import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, "models", "visual_analytics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define figure size (matching the wide aspect ratio of the user's image)
    fig_width = 8
    fig_height = 4
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.axis('off')
    
    # Precise, high-impact color palette matching the academic specifications
    header_bg = "#E6EFF8"      # Gentle light blue-gray for headers
    total_bg = "#E2E8F0"       # Neutral gray-blue for Total Latency highlighting
    border_color = "#7A92A8"   # Slate blue-gray borders
    text_color = "#0F172A"     # Slate-900 deep text
    
    # Set background to pure white
    fig.patch.set_facecolor('white')
    
    # Draw Title & Subtitle (centered with perfect vertical spacing)
    plt.text(0.5, 0.94, "TABLE III", fontsize=14, weight="bold", ha="center", va="center", color=text_color)
    plt.text(0.5, 0.86, "RUNTIME PERFORMANCE", fontsize=14, weight="bold", ha="center", va="center", color=text_color)
    
    # Table dimensions in normalized coordinates (0 to 1)
    x_start = 0.02
    width = 0.96
    y_start = 0.15
    row_height = 0.115
    
    # Row contents: (Operation, Avg Time, IsBold, BgColor)
    rows_data = [
        ("Operation", "Avg Time (ms)", True, header_bg),
        ("Pre-processing", "13.7ms", False, "#FFFFFF"),
        ("Inference (CPU)", "117.7ms", False, "#FFFFFF"),
        ("Local Visualization", "18.0ms", False, "#FFFFFF"),
        ("Total Latency", "149.4ms", True, total_bg)
    ]
    
    # Draw grid boxes and text
    for idx, (op, time_val, is_bold, bg) in enumerate(rows_data):
        # Calculate Y position in descending order so 0 is at the top
        y_pos = y_start + (4 - idx) * row_height
        
        # Draw background patch for the row
        rect = patches.Rectangle(
            (x_start, y_pos), width, row_height,
            facecolor=bg, edgecolor=border_color, linewidth=1.2, fill=True
        )
        ax.add_patch(rect)
        
        # Write left column text (Operation)
        font_weight = "bold" if is_bold else "normal"
        plt.text(
            x_start + width / 4, y_pos + row_height / 2, op,
            fontsize=12, weight=font_weight, ha="center", va="center", color=text_color
        )
        
        # Write right column text (Avg Time)
        plt.text(
            x_start + 3 * width / 4, y_pos + row_height / 2, time_val,
            fontsize=12, weight=font_weight, ha="center", va="center", color=text_color
        )
        
    # Draw vertical divider line down the exact middle of the table
    x_mid = x_start + width / 2
    y_top = y_start + 5 * row_height
    plt.plot([x_mid, x_mid], [y_start, y_top], color=border_color, linewidth=1.2)
    
    # Save the table
    plt.tight_layout()
    png_path = os.path.join(output_dir, "runtime_performance_table.png")
    pdf_path = os.path.join(output_dir, "runtime_performance_table.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.savefig(pdf_path, format="pdf", bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    print(f"Successfully generated runtime table image at:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")

if __name__ == "__main__":
    main()
