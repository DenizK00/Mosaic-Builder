import sys
import os, os.path
from PIL import Image, ImageOps, ImageEnhance
from multiprocessing import Process, Queue, cpu_count
import random
import math
import time  # Make sure to import the time module

# Improved configuration parameters
TILE_SIZE = 60  # Base tile size
TILE_MATCH_RES = 20  # Increased for more precise matching
ENLARGEMENT = 2
MIN_TILES = 10
DEFAULT_OPACITY = 0.3

# Enhanced tile processing parameters
SMALL_TILE_SCALE = 6  # Higher scale factor for better quality small tiles
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

            # Enhanced image processing
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)  # Increased color enhancement
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)  # Added sharpness enhancement

            # Get the largest square from the image
            w, h = img.size
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            # Create multiple variations with improved quality
            variations = []
            for angle in [0, 90, 180, 270]:
                rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC)
                variations.extend([
                    rotated,
                    ImageOps.mirror(rotated),
                    ImageOps.flip(rotated)
                ])

            img = random.choice(variations)

            # Create large tile with high quality
            large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            
            # Create small tile with enhanced resolution
            small_tile_size = int(TILE_SIZE * SMALL_TILE_SCALE / TILE_BLOCK_SIZE)
            small_tile_img = img.resize((small_tile_size, small_tile_size), Image.LANCZOS)
            
            # Apply additional enhancements to small tile
            enhancer = ImageEnhance.Sharpness(small_tile_img)
            small_tile_img = enhancer.enhance(1.2)
            
            # Removed the Detail enhancement as it does not exist
            # You can add other enhancements if needed, like Brightness or Contrast
            enhancer = ImageEnhance.Contrast(small_tile_img)
            small_tile_img = enhancer.enhance(1.1)  # Optional contrast enhancement

            return (large_tile_img.convert('RGB'), small_tile_img.convert('RGB'))
        except Exception as e:
            print(f"Error processing tile {tile_path}: {str(e)}")
            return (None, None)

    def get_tiles(self):
        large_tiles = []
        small_tiles = []
        tile_names = []
        usage_count = {}

        print('Reading tiles from {}...'.format(self.tiles_directory))

        # Enhanced recursive tile processing
        for root, _, files in os.walk(self.tiles_directory):
            for tile_name in files:
                if tile_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Added TIFF support
                    print('Reading {:40.40}'.format(tile_name), flush=True, end='\r')
                    tile_path = os.path.join(root, tile_name)
                    large_tile, small_tile = self.__process_tile(tile_path)
                    if large_tile and small_tile:  # Ensure both tiles are processed successfully
                        large_tiles.append(large_tile)
                        small_tiles.append(small_tile)
                        tile_names.append(tile_name)
                        usage_count[tile_name] = 0

        print('Processed {} tiles.'.format(len(large_tiles)))

        # Improved tile duplication with enhanced variations
        while len(large_tiles) < MIN_TILES:
            least_used_tiles = [tile for tile in range(len(tile_names)) 
                              if usage_count[tile_names[tile]] < 3]
            
            if not least_used_tiles:
                break

            idx = random.choice(least_used_tiles)
            large_tile = large_tiles[idx].copy()
            small_tile = small_tiles[idx].copy()

            # Apply enhanced variations
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.9, 1.1)
            color_factor = random.uniform(0.95, 1.05)

            # Apply enhancements to both large and small tiles
            for tile in [large_tile, small_tile]:
                enhancer = ImageEnhance.Brightness(tile)
                tile = enhancer.enhance(brightness_factor)
                
                enhancer = ImageEnhance.Contrast(tile)
                tile = enhancer.enhance(contrast_factor)
                
                enhancer = ImageEnhance.Color(tile)
                tile = enhancer.enhance(color_factor)

            large_tiles.append(large_tile)
            small_tiles.append(small_tile)
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
        # Store average brightness for each tile
        self.tile_brightness = [self._calculate_brightness(tile) for tile in tiles_data]
    
    def _calculate_brightness(self, tile_data):
        """Calculate the average brightness of a tile."""
        if not tile_data:
            return 0
        # Using the first pixel as a sample since tiles are already processed to be uniform
        total = sum(tile_data[0]) / 3  # Average of RGB
        return total / 255.0  # Normalize to 0-1
    
    def _get_tile_diff(self, t1, t2, bail_out_value):
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
    
    def get_best_fit_tiles(self, img_data, num_candidates=10):
        """Returns multiple good tile matches instead of just the best one."""
        differences = []
        target_brightness = self._calculate_brightness(img_data)
        
        for idx, tile_data in enumerate(self.tiles_data):
            # Calculate color difference
            color_diff = self._get_tile_diff(img_data, tile_data, sys.maxsize)
            # Calculate brightness difference
            brightness_diff = abs(target_brightness - self.tile_brightness[idx])
            
            # Combined score (weighted sum of color and brightness differences)
            total_diff = color_diff * 0.7 + (brightness_diff * 255) * 0.3
            differences.append((idx, total_diff))
        
        # Sort by difference and return top candidates
        differences.sort(key=lambda x: x[1])
        return differences[:num_candidates]


class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.original = original_img.copy()
        self.x_tile_count = int(original_img.size[0] / TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / TILE_SIZE)
        self.total_tiles = self.x_tile_count * self.y_tile_count
        self.recent_tiles = []  # Track recently used tiles
        self.max_recent = 50    # Number of recent tiles to track
    
    def select_tile(self, candidates, used_tiles):
        """Select a tile from candidates while maintaining variety."""
        # Filter out recently used tiles if possible
        available_candidates = [c for c in candidates if c[0] not in self.recent_tiles]
        
        if not available_candidates:
            available_candidates = candidates
        
        # Weight candidates by their ranking
        weights = []
        total_candidates = len(available_candidates)
        for i, (idx, diff) in enumerate(available_candidates):
            # Higher weight for better matches, but still give some chance to others
            weight = (total_candidates - i) ** 2
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Select tile using weights
        selected_idx = random.choices(range(len(available_candidates)), weights=weights, k=1)[0]
        tile_idx = available_candidates[selected_idx][0]
        
        # Update recent tiles
        self.recent_tiles.append(tile_idx)
        if len(self.recent_tiles) > self.max_recent:
            self.recent_tiles.pop(0)
        
        return tile_idx

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

    def get_average_color(self, tile_data):
        """Calculate the average color of a tile."""
        r, g, b = 0, 0, 0
        for pixel in tile_data:
            r += pixel[0]
            g += pixel[1]
            b += pixel[2]
        count = len(tile_data)
        return (r // count, g // count, b // count)

    def get_random_tile(self, all_tile_data_large, last_tile_index, used_tiles):
        """Select a random tile that is not the same as the last used tile and ensures all tiles are used."""
        tile_count = len(all_tile_data_large)
        
        # If all tiles have been used, reset the used_tiles list
        if len(used_tiles) == tile_count:
            used_tiles.clear()

        # Select a tile that hasn't been used yet
        available_tiles = [i for i in range(tile_count) if i not in used_tiles and i != last_tile_index]
        
        if available_tiles:
            new_tile_index = random.choice(available_tiles)
        else:
            # If all tiles have been used, allow reuse
            new_tile_index = last_tile_index

        # Add the selected tile to the used list
        used_tiles.append(new_tile_index)

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
    print("Starting mosaic generation...")
    mosaic = MosaicImage(original_img_large)
    active_workers = WORKER_COUNT
    used_tiles = set()
    
    while True:
        try:
            img_coords, candidates = result_queue.get()
            
            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                # Select tile using the new weighted selection system
                best_fit_tile_index = mosaic.select_tile(candidates, used_tiles)
                used_tiles.add(best_fit_tile_index)
                
                tile_data = all_tile_data_large[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords, opacity)
                
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
            
            # Get multiple candidate tiles instead of just one
            candidate_tiles = tile_fitter.get_best_fit_tiles(img_data)
            result_queue.put((img_coords, candidate_tiles))
        except KeyboardInterrupt:
            pass
    
    result_queue.put((EOQ_VALUE, EOQ_VALUE))


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
    # output_path = OUT_FILE
    # mosaic.save(output_path)
    # print('\nFinished, output is in', output_path)
    return mosaic.image  # Return the generated mosaic image


class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        print("Progress: {:04.1f}%".format(100 * self.counter / self.total), 
              flush=True, end='\r')
        

def mosaic(img_path, tiles_path, opacity=DEFAULT_OPACITY, resolution=(2400, 2400)):
    image_data = TargetImage(img_path).get_data()
    tiles_data = TileProcessor(tiles_path).get_tiles()
    if tiles_data[0]:
        mosaic_image = MosaicImage(image_data[0])  # Create the mosaic image object
        compose(image_data, tiles_data, opacity)  # Generate and save the mosaic image
        
        time.sleep(1)  # Wait for a short duration to ensure the image is fully generated
        
        # Load the saved image to display
        return Image.open(OUT_FILE)  # Return the generated mosaic image from the file
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