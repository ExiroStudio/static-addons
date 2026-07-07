import math
from PIL import Image, ImageDraw, ImageFont

# Character set to support
CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789><_ -!:"
GLYPH_SIZE = 16
HIGH_RES = 128  # High resolution for distance field computation
FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

def generate_glyph_sdf(char):
    # Create a high-res image
    img = Image.new("L", (HIGH_RES, HIGH_RES), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, int(HIGH_RES * 0.8))
    except IOError:
        font = ImageFont.load_default()
        
    # Get text size and draw centered
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Draw text centered in the high-res box
    x = (HIGH_RES - w) / 2 - bbox[0]
    y = (HIGH_RES - h) / 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    
    # Convert image to a list of lists of booleans (binary array)
    binary = []
    for y in range(HIGH_RES):
        row = []
        for x in range(HIGH_RES):
            row.append(img.getpixel((x, y)) > 127)
        binary.append(row)
        
    # Find boundary coordinates (where neighbors change value)
    edge_coords = []
    for y in range(1, HIGH_RES - 1):
        for x in range(1, HIGH_RES - 1):
            val = binary[y][x]
            if (binary[y-1][x] != val or
                binary[y+1][x] != val or
                binary[y][x-1] != val or
                binary[y][x+1] != val):
                edge_coords.append((y, x))
                
    # Grid coordinates for the 16x16 low-res target
    xs = [i * (HIGH_RES - 1) / (GLYPH_SIZE - 1) for i in range(GLYPH_SIZE)]
    ys = [i * (HIGH_RES - 1) / (GLYPH_SIZE - 1) for i in range(GLYPH_SIZE)]
    
    # Initialize flat list for the 16x16 matrix
    sdf_data = []
    
    max_radius = HIGH_RES / 6.0
    
    for row_idx, y_val in enumerate(ys):
        for col_idx, x_val in enumerate(xs):
            is_inside = binary[int(round(y_val))][int(round(x_val))]
            
            if not edge_coords:
                dist = -HIGH_RES
            else:
                # Find minimum distance to any edge pixel
                min_dist = float('inf')
                for ey, ex in edge_coords:
                    d = math.hypot(ey - y_val, ex - x_val)
                    if d < min_dist:
                        min_dist = d
                dist = min_dist if is_inside else -min_dist
            
            # Map distance to 0..255
            normalized = 128 + int(max(-127.0, min(127.0, dist / max_radius * 127.0)))
            sdf_data.append(normalized)
            
    return sdf_data

def main():
    packed_u32s = []
    
    print(f"Generating SDF for {len(CHARSET)} characters...")
    for char in CHARSET:
        flat = generate_glyph_sdf(char)
        # Pack 256 bytes into 64 u32s
        for i in range(0, len(flat), 4):
            b0 = flat[i]
            b1 = flat[i+1]
            b2 = flat[i+2]
            b3 = flat[i+3]
            val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            packed_u32s.append(val)
            
    # Output WGSL constant declaration
    wgsl_filename = "shaders/msdf_atlas.wgsl"
    import os
    os.makedirs("shaders", exist_ok=True)
    
    with open(wgsl_filename, "w") as f:
        f.write("// Auto-generated font atlas containing SDF data\n")
        f.write(f"// Character set: {CHARSET}\n")
        f.write(f"const CHARSET_LEN: u32 = {len(CHARSET)}u;\n")
        f.write(f"const GLYPH_SIZE: u32 = {GLYPH_SIZE}u;\n\n")
        f.write(f"const FONT_ATLAS = array<u32, {len(packed_u32s)}>(\n")
        for chunk in range(0, len(packed_u32s), 8):
            line = ", ".join(f"0x{val:08x}u" for val in packed_u32s[chunk:chunk+8])
            f.write(f"    {line},\n")
        f.write(");\n")
        
    print(f"Successfully generated {wgsl_filename}")

if __name__ == "__main__":
    main()
