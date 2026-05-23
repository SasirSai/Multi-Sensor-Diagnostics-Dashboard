import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def draw_rounded_rect(ax, x, y, w, h, text, bg, border, font_size=10, is_bold=True, text_color="black"):
    """Draw a beautifully styled rounded rectangle with centered text."""
    # Matplotlib FancyBboxPatch draws rounded boxes elegantly
    box = FancyBboxPatch(
        (x + 0.01, y + 0.01), w - 0.02, h - 0.02,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=bg, edgecolor=border, linewidth=1.2, fill=True
    )
    ax.add_patch(box)
    
    # Render centered text
    weight = "bold" if is_bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        fontsize=font_size, weight=weight, ha="center", va="center", color=text_color
    )

def draw_diamond(ax, x, y, w, h, text, bg, border, font_size=10, text_color="black"):
    """Draw a sharp decision diamond with centered text."""
    # Coordinate offsets for a perfect diamond
    # Midpoints of the rectangular box
    x_mid = x + w / 2
    y_mid = y + h / 2
    
    points = [
        (x_mid, y + h),      # Top vertex
        (x + w, y_mid),      # Right vertex
        (x_mid, y),          # Bottom vertex
        (x, y_mid)           # Left vertex
    ]
    
    diamond = Polygon(points, facecolor=bg, edgecolor=border, linewidth=1.2, fill=True)
    ax.add_patch(diamond)
    
    # Render centered text
    ax.text(
        x_mid, y_mid, text,
        fontsize=font_size, weight="bold", ha="center", va="center", color=text_color
    )

def draw_arrow(ax, x1, y1, x2, y2, text="", text_pos="side"):
    """Draw a clean diagnostic flow arrow with optional labels."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12)
    )
    
    # Render label along the path if provided
    if text:
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        if text_pos == "side":
            ax.text(x_mid + 0.01, y_mid, text, fontsize=9, weight="bold", ha="left", va="center")
        elif text_pos == "above":
            ax.text(x_mid, y_mid + 0.01, text, fontsize=9, weight="bold", ha="center", va="bottom")
        elif text_pos == "below":
            ax.text(x_mid, y_mid - 0.02, text, fontsize=9, weight="bold", ha="center", va="top")


# =============================================================================
# FLOWCHART 1: SYSTEM ARCHITECTURE
# =============================================================================

def generate_system_architecture(output_dir):
    print("Generating Flowchart 1: System Architecture...")
    fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Lavender theme matching Flowchart 1
    bg_color = "#E8EAF6"      # Soft lavender/blue
    border_color = "#1A237E"  # Indigo-900 border
    
    box_w, box_h = 0.5, 0.075
    x_pos = 0.25
    
    # Vertically stacked layers
    y_positions = [0.85, 0.68, 0.51, 0.34, 0.17]
    labels = [
        "Sensors (Vib/Acous/Elec)",
        "DAQ / ESP32 Gateway",
        "Edge CPU (Raspberry Pi/Jetson)",
        "RF Inference + SHAP Engine",
        "Safety Controller (PLC)"
    ]
    
    # Draw boxes
    for y, text in zip(y_positions, labels):
        draw_rounded_rect(ax, x_pos, y, box_w, box_h, text, bg_color, border_color, font_size=11)
        
    # Draw vertical arrows
    for i in range(len(y_positions) - 1):
        y_from = y_positions[i]
        y_to = y_positions[i+1] + box_h
        draw_arrow(ax, x_pos + box_w/2, y_from, x_pos + box_w/2, y_to)
        
    # Draw side branch to XAI Dashboard
    # Rightward box next to RF Inference
    draw_rounded_rect(ax, 0.70, y_positions[3], 0.25, box_h, "XAI Dashboard", bg_color, border_color, font_size=11)
    draw_arrow(ax, x_pos + box_w, y_positions[3] + box_h/2, 0.70, y_positions[3] + box_h/2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_system_architecture.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "flowchart_system_architecture.pdf"), format="pdf", bbox_inches='tight')
    plt.close()


# =============================================================================
# FLOWCHART 2: DIAGNOSTIC PIPELINE
# =============================================================================

def generate_diagnostic_pipeline(output_dir):
    print("Generating Flowchart 2: Data Flow Diagnostic Pipeline...")
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Styles matching color schemes of Flowchart 2
    orange_bg = "#FFEDD5"   # Light orange
    orange_border = "#EA580C"
    
    green_bg = "#DCFCE7"    # Light green
    green_border = "#16A34A"
    
    blue_bg = "#DBEAFE"     # Light blue
    blue_border = "#2563EB"
    
    pink_bg = "#FCE7F3"     # Light pink
    pink_border = "#DB2777"
    
    red_bg = "#FEE2E2"      # Light red
    red_border = "#DC2626"
    
    # 1. Inputs (stacked on the left)
    inputs_y = [0.70, 0.43, 0.16]
    input_w, input_h = 0.17, 0.15
    input_labels = [
        "Accelerometer\n(Vibration / IMU)",
        "Microphone\n(Acoustic Emissions)",
        "CT and Thermocouple\n(Current / Temp)"
    ]
    for y, label in zip(inputs_y, input_labels):
        draw_rounded_rect(ax, 0.02, y, input_w, input_h, label, orange_bg, orange_border, font_size=9)
        
    # 2. DAQ Synchronisation
    draw_rounded_rect(ax, 0.24, 0.43, 0.16, 0.15, "DAQ Synchronisation", green_bg, green_border, font_size=10)
    
    # Draw arrows from inputs to DAQ Synchronisation
    draw_arrow(ax, 0.02 + input_w, 0.70 + input_h/2, 0.24, 0.43 + 0.11)
    draw_arrow(ax, 0.02 + input_w, 0.43 + input_h/2, 0.24, 0.43 + input_h/2)
    draw_arrow(ax, 0.02 + input_w, 0.16 + input_h/2, 0.24, 0.43 + 0.04)
    
    # 3. Feature Extraction
    draw_rounded_rect(ax, 0.44, 0.43, 0.16, 0.15, "Feature Extraction\n(43-Dim Vector)", blue_bg, blue_border, font_size=10)
    draw_arrow(ax, 0.24 + 0.16, 0.43 + 0.15/2, 0.44, 0.43 + 0.15/2)
    
    # 4. Classifier
    draw_rounded_rect(ax, 0.64, 0.43, 0.15, 0.15, "Random Forest\nClassifier", orange_bg, orange_border, font_size=10)
    draw_arrow(ax, 0.44 + 0.16, 0.43 + 0.15/2, 0.64, 0.43 + 0.15/2)
    
    # 5. SHAP side branch
    draw_rounded_rect(ax, 0.64, 0.70, 0.15, 0.15, "SHAP Explainability\n(XAI Dashboard)", pink_bg, pink_border, font_size=9)
    draw_arrow(ax, 0.64 + 0.15/2, 0.43 + 0.15, 0.64 + 0.15/2, 0.70)
    
    # 6. Outputs (stacked on the right)
    output_w, output_h = 0.17, 0.15
    outputs_y = [0.70, 0.43, 0.16]
    output_labels = [
        "Normal Operation\n(No Intervention)",
        "Reduce Motor Torque\n(Alignment Mode)",
        "Emergency Stop\n(BPFI / BPFO)"
    ]
    output_bgs = [green_bg, blue_bg, red_bg]
    output_borders = [green_border, blue_border, red_border]
    
    for y, label, bg, border in zip(outputs_y, output_labels, output_bgs, output_borders):
        draw_rounded_rect(ax, 0.81, y, output_w, output_h, label, bg, border, font_size=9)
        
    # Draw output labels and arrows
    draw_arrow(ax, 0.64 + 0.15, 0.43 + 0.11, 0.81, 0.70 + output_h/2, "Normal", "below")
    draw_arrow(ax, 0.64 + 0.15, 0.43 + 0.15/2, 0.81, 0.43 + output_h/2, "Moderate", "below")
    draw_arrow(ax, 0.64 + 0.15, 0.43 + 0.04, 0.81, 0.16 + output_h/2, "High Severity", "below")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_diagnostic_pipeline.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "flowchart_diagnostic_pipeline.pdf"), format="pdf", bbox_inches='tight')
    plt.close()


# =============================================================================
# FLOWCHART 3: DECISION LOGIC
# =============================================================================

def generate_decision_logic(output_dir):
    print("Generating Flowchart 3: Decision Logic...")
    fig, ax = plt.subplots(figsize=(6, 9), dpi=300)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Colors matching Flowchart 3
    gray_bg = "#F8FAFC"      # Neutral slate background
    gray_border = "#475569"
    
    yellow_bg = "#FEF08A"    # Decision diamond yellow
    yellow_border = "#CA8A04"
    
    orange_bg = "#FFEDD5"    # Muted orange warning
    orange_border = "#EA580C"
    
    red_bg = "#FEE2E2"       # Emergency Stop Red
    red_border = "#DC2626"
    
    # Y Coordinates
    y_start = 0.90
    y_dist = 0.18
    w, h = 0.35, 0.075
    x_pos = 0.15
    
    # 1. Start rounded rectangle
    draw_rounded_rect(ax, x_pos, y_start, w, h, "AI Inference Triggered", gray_bg, gray_border, font_size=11)
    
    # 2. Decision Diamond 1: Anomaly Detected?
    y_pos = y_start - y_dist
    draw_diamond(ax, x_pos, y_pos, w, h, "Anomaly Detected?", yellow_bg, yellow_border, font_size=10)
    draw_arrow(ax, x_pos + w/2, y_start, x_pos + w/2, y_pos + h)
    
    # No branch to Continue Ops
    draw_rounded_rect(ax, 0.60, y_pos, 0.28, h, "Continue Ops", gray_bg, gray_border, font_size=11)
    draw_arrow(ax, x_pos + w, y_pos + h/2, 0.60, y_pos + h/2, "No", "above")
    
    # 3. Decision Diamond 2: Sensor Integrity OK?
    y_pos2 = y_pos - y_dist
    draw_diamond(ax, x_pos, y_pos2, w, h, "Sensor Integrity OK?", yellow_bg, yellow_border, font_size=9.5)
    draw_arrow(ax, x_pos + w/2, y_pos, x_pos + w/2, y_pos2 + h, "Yes", "side")
    
    # No branch to Sensor Fault Alarm
    draw_rounded_rect(ax, 0.60, y_pos2, 0.28, h, "Sensor Fault Alarm", orange_bg, orange_border, font_size=10)
    draw_arrow(ax, x_pos + w, y_pos2 + h/2, 0.60, y_pos2 + h/2, "No", "above")
    
    # 4. Decision Diamond 3: Anomaly Persists > 3s?
    y_pos3 = y_pos2 - y_dist
    draw_diamond(ax, x_pos, y_pos3, w, h, "Anomaly Persists > 3s?", yellow_bg, yellow_border, font_size=9.5)
    draw_arrow(ax, x_pos + w/2, y_pos2, x_pos + w/2, y_pos3 + h, "Yes", "side")
    
    # No branch to Log Transient
    draw_rounded_rect(ax, 0.60, y_pos3, 0.28, h, "Log Transient", gray_bg, gray_border, font_size=11)
    draw_arrow(ax, x_pos + w, y_pos3 + h/2, 0.60, y_pos3 + h/2, "No", "above")
    
    # 5. Decision Diamond 4: Severity > 80%?
    y_pos4 = y_pos3 - y_dist
    draw_diamond(ax, x_pos, y_pos4, w, h, "Severity > 80%?", yellow_bg, yellow_border, font_size=10)
    draw_arrow(ax, x_pos + w/2, y_pos3, x_pos + w/2, y_pos4 + h, "Yes", "side")
    
    # No branch to Reduce Torque
    draw_rounded_rect(ax, 0.60, y_pos4, 0.28, h, "Reduce Torque", orange_bg, orange_border, font_size=11)
    draw_arrow(ax, x_pos + w, y_pos4 + h/2, 0.60, y_pos4 + h/2, "No", "above")
    
    # 6. EMERGENCY STOP red box
    y_pos5 = y_pos4 - y_dist
    draw_rounded_rect(ax, x_pos, y_pos5, w, h, "EMERGENCY STOP", red_bg, red_border, font_size=11, text_color=red_border)
    draw_arrow(ax, x_pos + w/2, y_pos4, x_pos + w/2, y_pos5 + h, "Yes", "side")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_decision_logic.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "flowchart_decision_logic.pdf"), format="pdf", bbox_inches='tight')
    plt.close()


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, "models", "visual_analytics")
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Draw Academic Flowcharts ---")
    generate_system_architecture(output_dir)
    generate_diagnostic_pipeline(output_dir)
    generate_decision_logic(output_dir)
    print("--- Flowcharts Exported Successfully ---")

if __name__ == "__main__":
    main()
