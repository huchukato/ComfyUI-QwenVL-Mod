#!/usr/bin/env python3
"""
Create a custom icon for QwenVL-Mod nodes
"""

from PIL import Image, ImageDraw, ImageFont
import base64
import io

def create_qwenvl_mod_icon():
    """Create a custom icon for QwenVL-Mod"""
    # Create 64x64 image with transparent background
    size = 64
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Use a nice blue color for QwenVL-Mod
    primary_color = (41, 128, 185)  # Blue
    secondary_color = (52, 152, 219)  # Light blue
    
    # Draw rounded rectangle background
    margin = 8
    draw.rounded_rectangle(
        [margin, margin, size-margin, size-margin],
        radius=12,
        fill=primary_color
    )
    
    # Draw inner rectangle
    inner_margin = 12
    draw.rounded_rectangle(
        [inner_margin, inner_margin, size-inner_margin, size-inner_margin],
        radius=8,
        fill=secondary_color
    )
    
    # Try to use a font, fallback to default if not available
    try:
        # Try different font options
        font_options = [
            "Arial.ttf",
            "Helvetica.ttf", 
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        
        font = None
        for font_path in font_options:
            try:
                font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw text "QVL"
    text = "QVL"
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 2
    
    # Draw text with shadow effect
    shadow_offset = 1
    draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0, 128), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img

def icon_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def main():
    """Create and save the icon"""
    print("🎨 Creating QwenVL-Mod icon...")
    
    # Create icon
    icon = create_qwenvl_mod_icon()
    
    # Convert to base64
    icon_b64 = icon_to_base64(icon)
    
    # Save to file
    with open('qwenvl_mod_icon.txt', 'w') as f:
        f.write(icon_b64)
    
    print(f"✅ Icon created and saved to qwenvl_mod_icon.txt")
    print(f"📏 Size: 64x64 pixels")
    print(f"📝 Base64 length: {len(icon_b64)} characters")
    print(f"🔗 Ready to use in node code!")
    
    # Also save as PNG for preview
    icon.save('qwenvl_mod_icon.png')
    print(f"🖼️ Preview saved as qwenvl_mod_icon.png")

if __name__ == "__main__":
    main()
