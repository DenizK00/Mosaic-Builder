import sys
import os, os.path
from PIL import Image, ImageOps, ImageEnhance
from multiprocessing import Process, Queue, cpu_count
import random
import math

# Improved configuration parameters
TILE_SIZE = 50  # Increased for better visibility of individual tiles
TILE_MATCH_RES = 5  # Increased for better matching
ENLARGEMENT = 2  # Kept the same
MIN_TILES = 10  # Minimum number of tiles needed
DEFAULT_OPACITY = 0.3  # Reduced default opacity for more subtle effect

# Auto-calculate tile counts based on image size
TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
WORKER_COUNT = max(cpu_count() - 1, 1)
OUT_FILE = 'mosaic.jpeg'
EOQ_VALUE = None

class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory
        
    def __process_tile(self, tile_path):
        try:
            img = Image.open(tile_path)
            img = ImageOps.exif_transpose(img)

            # Enhance the tile image
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # Slightly increase contrast
            
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.1)  # Slightly increase color saturation

            # Get the largest square from the image
            w, h = img.size
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            # Create multiple variations of each tile
            variations = []
            for angle in [0, 90, 180, 270]:
                rotated = img.rotate(angle, expand=True)
                # Add both normal and flipped versions
                variations.extend([
                    rotated,
                    ImageOps.mirror(rotated),
                    ImageOps.flip(rotated)
                ])

            # Randomly select a variation
            img = random.choice(variations)

            large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            small_tile_img = img.resize((int(TILE_SIZE/TILE_BLOCK_SIZE), 
                                       int(TILE_SIZE/TILE_BLOCK_SIZE)), Image.LANCZOS)

            return (large_tile_img.convert('RGB'), small_tile_img.convert('RGB'))
        except Exception as e:
            print(f"Error processing tile {tile_path}: {str(e)}")
            return (None, None)

    def get_tiles(self):
        large_tiles = []
        small_tiles = []
        tile_names = []  # List to store tile names
        usage_count = {}  # Dictionary to track usage of each tile

        print('Reading tiles from {}...'.format(self.tiles_directory))

        # Search the tiles directory recursively
        for root, _, files in os.walk(self.tiles_directory):
            for tile_name in files:
                if tile_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    print('Reading {:40.40}'.format(tile_name), flush=True, end='\r')
                    tile_path = os.path.join(root, tile_name)
                    large_tile, small_tile = self.__process_tile(tile_path)
                    if large_tile:
                        large_tiles.append(large_tile)
                        small_tiles.append(small_tile)
                        tile_names.append(tile_name)  # Store the tile name
                        usage_count[tile_name] = 0  # Initialize usage count for the tile

        print('Processed {} tiles.'.format(len(large_tiles)))

        # If we have fewer tiles than MIN_TILES, duplicate them with variations
        while len(large_tiles) < MIN_TILES:
            # Select a tile that has been used the least
            least_used_tiles = [tile for tile in range(len(tile_names)) if usage_count[tile_names[tile]] < 3]  # Limit to 3 uses
            if not least_used_tiles:
                break  # Exit if all tiles have been used too many times

            idx = random.choice(least_used_tiles)
            large_tile = large_tiles[idx].copy()
            small_tile = small_tiles[idx].copy()

            # Apply random enhancement to create variation
            enhancer = ImageEnhance.Brightness(large_tile)
            large_tile = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Brightness(small_tile)
            small_tile = enhancer.enhance(random.uniform(0.8, 1.2))

            large_tiles.append(large_tile)
            small_tiles.append(small_tile)

            # Update usage count
            usage_count[tile_names[idx]] += 1

        return (large_tiles, small_tiles)

class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        
        # Enhance the target image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Calculate dimensions to maintain aspect ratio
        target_width = 1200  # Maximum width
        w_ratio = target_width / img.size[0]
        new_height = int(img.size[1] * w_ratio)
        
        w = target_width * ENLARGEMENT
        h = new_height * ENLARGEMENT
        
        large_img = img.resize((w, h), Image.LANCZOS)
        w_diff = (w % TILE_SIZE)/2
        h_diff = (h % TILE_SIZE)/2
        
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize((int(w/TILE_BLOCK_SIZE), int(h/TILE_BLOCK_SIZE)), 
                                   Image.LANCZOS)

        return (large_img.convert('RGB'), small_img.convert('RGB'))


class TileFitter:
    def __init__(self, tiles_data):
        self.tiles_data = tiles_data

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            # Improved color difference calculation using weighted RGB
            r_diff = (t1[i][0] - t2[i][0]) * 0.3
            g_diff = (t1[i][1] - t2[i][1]) * 0.59
            b_diff = (t1[i][2] - t2[i][2]) * 0.11
            diff += (r_diff**2 + g_diff**2 + b_diff**2)
            if diff > bail_out_value:
                return diff
        return diff
    
    def get_best_fit_tile(self, img_data):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = 0

        for tile_data in self.tiles_data:
            diff = self.__get_tile_diff(img_data, tile_data, min_diff)
            if diff < min_diff:
                min_diff = diff
                best_fit_tile_index = tile_index
            tile_index += 1

        return best_fit_tile_index, min_diff
    

class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.original = original_img.copy()  # Keep a copy of the original
        self.x_tile_count = int(original_img.size[0] / TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / TILE_SIZE)
        self.total_tiles = self.x_tile_count * self.y_tile_count
        self.last_tile_used = {}  # Dictionary to track the last tile used at each position

    def add_tile(self, tile_data, coords, opacity=DEFAULT_OPACITY):
        # Create tile image
        img = Image.new('RGB', (TILE_SIZE, TILE_SIZE))
        img.putdata(tile_data)
        
        # Create alpha channel with lower opacity
        alpha = Image.new('L', img.size, int(opacity * 255))
        img.putalpha(alpha)
        
        # Get the corresponding section of the original image
        orig_section = self.original.crop(coords)
        
        # Create a new image for blending
        blend_image = Image.new('RGBA', img.size)
        
        # Paste original section with higher opacity
        orig_alpha = Image.new('L', img.size, int((1 - opacity * 0.5) * 255))
        orig_section.putalpha(orig_alpha)
        blend_image.paste(orig_section, (0, 0))
        
        # Blend with the mosaic tile
        blend_image.alpha_composite(img)
        
        # Paste the blended result
        self.image.paste(blend_image, coords)

    def get_random_tile(self, all_tile_data_large, last_tile_index):
        """Select a random tile that is not the same as the last used tile."""
        tile_count = len(all_tile_data_large)
        new_tile_index = last_tile_index
        
        # Ensure the new tile is different from the last one
        while new_tile_index == last_tile_index:
            new_tile_index = random.randint(0, tile_count - 1)
        
        return new_tile_index

    def save(self, path):
        # Final enhancement with subtle adjustments
        enhancer = ImageEnhance.Sharpness(self.image)
        self.image = enhancer.enhance(1.1)  # Reduced sharpness enhancement
        
        # Slightly boost contrast
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(1.05)
        
        self.image.save(path, quality=95)


def build_mosaic(result_queue, all_tile_data_large, original_img_large, opacity):
    mosaic = MosaicImage(original_img_large)
    active_workers = WORKER_COUNT
    
    last_tile_index = None  # Track the last tile index used

    while True:
        try:
            img_coords, best_fit_tile_index, fit_quality = result_queue.get()

            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                # Use the provided opacity
                # Get a random tile that is not the same as the last used tile
                best_fit_tile_index = mosaic.get_random_tile(all_tile_data_large, last_tile_index)
                tile_data = all_tile_data_large[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords, opacity)

                # Update the last tile index used
                last_tile_index = best_fit_tile_index

        except KeyboardInterrupt:
            pass

    mosaic.save(OUT_FILE)
    print('\nFinished, output is in', OUT_FILE)


def fit_tiles(work_queue, result_queue, tiles_data):
    tile_fitter = TileFitter(tiles_data)

    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                break
            tile_index, fit_quality = tile_fitter.get_best_fit_tile(img_data)
            result_queue.put((img_coords, tile_index, fit_quality))
        except KeyboardInterrupt:
            pass

    result_queue.put((EOQ_VALUE, EOQ_VALUE, EOQ_VALUE))


def compose(original_img, tiles, opacity=DEFAULT_OPACITY):
    print('Building mosaic, press Ctrl-C to abort...')
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles

    all_tile_data_large = [list(tile.getdata()) for tile in tiles_large]
    all_tile_data_small = [list(tile.getdata()) for tile in tiles_small]

    work_queue = Queue(WORKER_COUNT)
    result_queue = Queue()

    mosaic = MosaicImage(original_img_large)  # Create the mosaic image object

    try:
        Process(target=build_mosaic, 
                args=(result_queue, all_tile_data_large, original_img_large, opacity)).start()

        for _ in range(WORKER_COUNT):
            Process(target=fit_tiles, 
                   args=(work_queue, result_queue, all_tile_data_small)).start()

        total_tiles = int(original_img_large.size[0] / TILE_SIZE) * int(original_img_large.size[1] / TILE_SIZE)
        progress = ProgressCounter(total_tiles)

        for x in range(int(original_img_large.size[0] / TILE_SIZE)):
            for y in range(int(original_img_large.size[1] / TILE_SIZE)):
                large_box = (x * TILE_SIZE, y * TILE_SIZE, 
                           (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
                small_box = (x * TILE_SIZE/TILE_BLOCK_SIZE, 
                           y * TILE_SIZE/TILE_BLOCK_SIZE,
                           (x + 1) * TILE_SIZE/TILE_BLOCK_SIZE, 
                           (y + 1) * TILE_SIZE/TILE_BLOCK_SIZE)
                work_queue.put((list(original_img_small.crop(small_box).getdata()), 
                              large_box))
                progress.update()

    except KeyboardInterrupt:
        print('\nHalting, saving partial image please wait...')

    finally:
        for _ in range(WORKER_COUNT):
            work_queue.put((EOQ_VALUE, EOQ_VALUE))

    # Save the mosaic image and return it
    output_path = OUT_FILE
    mosaic.save(output_path)
    print('\nFinished, output is in', output_path)
    return mosaic.image  # Return the generated mosaic image


class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        print("Progress: {:04.1f}%".format(100 * self.counter / self.total), 
              flush=True, end='\r')
        

def mosaic(img_path, tiles_path, opacity=DEFAULT_OPACITY):
    image_data = TargetImage(img_path).get_data()
    tiles_data = TileProcessor(tiles_path).get_tiles()
    if tiles_data[0]:
        mosaic_image = MosaicImage(image_data[0])  # Create the mosaic image object
        compose(image_data, tiles_data, opacity)  # Build the mosaic
        return mosaic_image.image  # Return the generated mosaic image
    else:
        print("ERROR: No images found in tiles directory '{}'".format(tiles_path))
        return None


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('ERROR: Usage: {} <image> <tiles directory>\r'.format(sys.argv[0]))
    else:
        source_image = sys.argv[1]
        tile_dir = sys.argv[2]
        if not os.path.isfile(source_image):
            print("ERROR: Unable to find image file '{}'".format(source_image))
        elif not os.path.isdir(tile_dir):
            print("ERROR: Unable to find tile directory '{}'".format(tile_dir))
        else:
            mosaic(source_image, tile_dir)