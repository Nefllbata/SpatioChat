import matplotlib.pyplot as plt
import numpy as np

def plot_spatial_layout(script_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    room_x, room_y, room_z = script_data.get('roomsize', [6.0, 6.0, 3.0])
    listener_pos = script_data.get('listener') or script_data.get('listener_pos')
    if listener_pos:
        lx, ly = (listener_pos[0], listener_pos[1])
        plot_lx, plot_ly = (-ly, lx)
        ax.scatter(plot_lx, plot_ly, c='red', marker='^', s=200, label='Listener (Facing Front)')
        ax.arrow(plot_lx, plot_ly, 0, 0.5, head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.7)
    for spk in script_data.get('speaker', []):
        sx, sy = (spk['pos'][0], spk['pos'][1])
        name = spk['name']
        plot_sx, plot_sy = (-sy, sx)
        ax.scatter(plot_sx, plot_sy, c='blue', s=150)
        ax.text(plot_sx, plot_sy + 0.2, name, fontsize=11, ha='center', va='bottom', fontweight='bold')
        if listener_pos:
            ax.plot([plot_lx, plot_sx], [plot_ly, plot_sy], 'k--', alpha=0.3)
            dx = sx - lx
            dy = sy - ly
            angle = np.degrees(np.arctan2(dy, dx))
            distance = np.sqrt(dx ** 2 + dy ** 2)
            mid_x = (plot_lx + plot_sx) / 2
            mid_y = (plot_ly + plot_sy) / 2
            ax.text(mid_x, mid_y, f'{angle:.1f}°\n{distance:.1f}m', fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax.set_xlim(-room_y - 1, 1)
    ax.set_ylim(-1, room_x + 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title('Spatial Dialogue Layout (AES69 Standard: +X=Front, +Y=Left)', pad=20, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.text(0.5, 0.95, 'FRONT (+X)', transform=ax.transAxes, ha='center', fontsize=12, alpha=0.5)
    ax.text(0.5, 0.05, 'BACK (-X)', transform=ax.transAxes, ha='center', fontsize=12, alpha=0.5)
    ax.text(0.05, 0.5, 'LEFT (+Y)', transform=ax.transAxes, va='center', rotation=90, fontsize=12, alpha=0.5)
    ax.text(0.95, 0.5, 'RIGHT (-Y)', transform=ax.transAxes, va='center', rotation=-90, fontsize=12, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
