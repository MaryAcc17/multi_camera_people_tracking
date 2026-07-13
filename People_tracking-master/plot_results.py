import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

def load_and_preprocess(file_path, time_window=None):
    """
    Load and preprocess tracking data with optional time window filtering.
    
    Args:
        file_path: Path to the CSV file
        time_window: Optional tuple of (start_time, end_time) in seconds
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = df['stamp_sec'] + df['stamp_nanosec'] * 1e-9
    df['timestamp'] -= df['timestamp'].min()
    
    if time_window is not None:
        start_time, end_time = time_window
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        
        df['timestamp'] -= start_time
        
    return df

def plot_tracking_comparison(data_files, assignment_dicts, method_names, time_window=None, output_path=None):
    """
    Plot tracking results comparison for all methods.
    
    Args:
        data_files: List of paths to tracking result files
        assignment_dicts: List of dictionaries mapping GT IDs to tracker IDs for each method
        method_names: List of method names for legend
        time_window: Optional tuple of (start_time, end_time) in seconds
        output_path: Optional path to save the figure
    """
    components = ['x', 'y', 'yaw', 'vx', 'vy']
    component_labels = {
        'x': 'X Position (m)',
        'y': 'Y Position (m)',
        'yaw': 'Yaw (rad)',
        'vx': 'X Velocity (m/s)',
        'vy': 'Y Velocity (m/s)'
    }
    
    gt_ids = set()
    for assign_dict in assignment_dicts:
        gt_ids.update(assign_dict.keys())
    
    
    method_styles = {
        0: {'linestyle': ':', 'alpha': 0.7, 'linewidth': 1.5},
        1: {'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.5, 'color':'blue'},
        2: {'linestyle': '-.', 'alpha': 0.7, 'linewidth': 1.5,'color':'darkorange'}
    }
    
    plt.style.use('seaborn-v0_8-paper')
    
    for gt_id in gt_ids:
        # Create figure
        fig, axs = plt.subplots(len(components), 1, 
                               figsize=(8, 10),
                               sharex=True,
                               gridspec_kw={'height_ratios': [1]*len(components)})
        
        # Track legend entries
        legend_elements = []
        
        # Plot ground truth and estimates for each component
        for comp_idx, (component, ax) in enumerate(zip(components, axs)):
            # Plot each method's results
            for method_idx, (file_path, assignment_dict, method_name) in enumerate(
                    zip(data_files, assignment_dicts, method_names)):
                
                df = load_and_preprocess(file_path, time_window)
                
                if df.empty:
                    print(f"Warning: No data in specified time window for {method_name}")
                    continue
                
                # Plot ground truth (only once, using first method's data)
                if method_idx == 0 and gt_id in assignment_dict:
                    vicon_id = int(gt_id[1:])
                    gt_data = df[(df['source'] == 'vicon') & (df['id'] == vicon_id)]
                    if not gt_data.empty:
                        ax.plot(gt_data['timestamp'], gt_data[component],
                               color='black', linestyle='-', linewidth=1.2)
                        if comp_idx == 0:  # Add to legend only once
                            legend_elements.append(Line2D([0], [0], color='black', linestyle='-', 
                                                        label='Ground Truth'))
                
                # Plot estimated trajectories
                if gt_id in assignment_dict:
                    for track_id in assignment_dict[gt_id]:
                        est_data = df[(df['source'] == 'tracker') & (df['id'] == track_id)]
                        if not est_data.empty:
                            line = ax.plot(est_data['timestamp'], est_data[component],
                                         **method_styles[method_idx])[0]
                            
                            # Add to legend only once (for first component)
                            if comp_idx == 0:
                                label = f'{method_name}'
                                if len(assignment_dict[gt_id]) > 1:
                                    label += f' (ID: {track_id})'
                                legend_elements.append(Line2D([0], [0], 
                                                            color=line.get_color(),
                                                            linestyle=method_styles[method_idx]['linestyle'],
                                                            linewidth=method_styles[method_idx].get('linewidth', 1.5),
                                                            label=label))
            
            # Customize each subplot
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(component_labels[component])
            if comp_idx == len(components) - 1:
                ax.set_xlabel('Time (s)')
        
        # Add legend to the first subplot
        axs[0].legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(0.98, 0.98), fontsize='small',
                     ncol=1)
        
        # Add title including time window information if specified
        title = f'Tracking Results Comparison - Agent {gt_id}'
        if time_window:
            title += f'\nTime Window: {time_window[0]:.1f}s to {time_window[1]:.1f}s'
        # fig.suptitle(title, y=0.92)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save as PNG with high DPI
        if output_path:
            time_suffix = f"_t{time_window[0]:.1f}-{time_window[1]:.1f}" if time_window else ""
            save_path = f"{output_path}/comparison_{gt_id}{time_suffix}.png"
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        else:
            plt.show()
        plt.close()

def create_plot(components, component_labels, n_rows, figsize=(8, 4)):
    """Create figure and axes with consistent styling."""
    fig, axs = plt.subplots(n_rows, 1, 
                           figsize=figsize,
                           sharex=True,
                           gridspec_kw={'height_ratios': [1]*n_rows})
    
    
    if n_rows == 1:
        axs = np.array([axs])
        
    return fig, axs

def plot_tracking_comparison_separated(data_files, assignment_dicts, method_names, time_window=None, output_path=None):
    """
    Plot tracking results comparison for all methods in three separate plots:
    position, velocity, and orientation.
    """
    
    plot_groups = {
        'position': {
            'components': ['x', 'y'],
            'labels': {
                'x': 'X Position (m)',
                'y': 'Y Position (m)'
            },
            'figsize': (8, 6)
        },
        'velocity': {
            'components': ['vx', 'vy'],
            'labels': {
                'vx': 'X Velocity (m/s)',
                'vy': 'Y Velocity (m/s)'
            },
            'figsize': (8, 6)
        },
        'orientation': {
            'components': ['yaw'],
            'labels': {
                'yaw': 'Yaw (rad)'
            },
            'figsize': (8, 3)
        }
    }
    
    
    method_styles = {
        0: {'linestyle': ':', 'alpha': 0.9, 'linewidth': 1.2},
        1: {'linestyle': '--', 'alpha': 0.9, 'linewidth': 1.2},
        2: {'linestyle': '-.', 'alpha': 0.9, 'linewidth': 1.2,'color': 'mediumorchid'}
    }
    
    plt.style.use('seaborn-v0_8-paper')
    
    # Get all unique GT IDs
    gt_ids = set()
    for assign_dict in assignment_dicts:
        gt_ids.update(assign_dict.keys())
    
    for gt_id in gt_ids:
        
        for plot_type, plot_info in plot_groups.items():
            components = plot_info['components']
            component_labels = plot_info['labels']
            
           
            fig, axs = create_plot(components, component_labels, 
                                 len(components), plot_info['figsize'])
            
            # Track legend entries
            legend_elements = []
            
            # Plot each component
            for comp_idx, (component, ax) in enumerate(zip(components, axs)):
                # Plot each method's results
                for method_idx, (file_path, assignment_dict, method_name) in enumerate(
                        zip(data_files, assignment_dicts, method_names)):
                    
                    df = load_and_preprocess(file_path, time_window)
                    
                    if df.empty:
                        print(f"Warning: No data in specified time window for {method_name}")
                        continue
                    
                    # Plot ground truth
                    if method_idx == 0 and gt_id in assignment_dict:
                        vicon_id = int(gt_id[1:])
                        gt_data = df[(df['source'] == 'vicon') & (df['id'] == vicon_id)]
                        if not gt_data.empty:
                            ax.plot(gt_data['timestamp'], gt_data[component],
                                   color='black', linestyle='-', linewidth=1.2, alpha=0.7)
                            if comp_idx == 0:  
                                legend_elements.append(Line2D([0], [0], color='black', 
                                                           linestyle='-', 
                                                           label='Ground Truth'))
                    
                    # Plot estimated trajectories
                    if gt_id in assignment_dict:
                        for track_id in assignment_dict[gt_id]:
                            est_data = df[(df['source'] == 'tracker') & (df['id'] == track_id)]
                            if not est_data.empty:
                                line = ax.plot(est_data['timestamp'], est_data[component],
                                             **method_styles[method_idx])[0]
                                
                                
                                if comp_idx == 0:
                                    label = f'{method_name}'
                                    if len(assignment_dict[gt_id]) > 1:
                                        label += f' (ID: {track_id})'
                                    legend_elements.append(Line2D([0], [0], 
                                                                color=line.get_color(),
                                                                linestyle=method_styles[method_idx]['linestyle'],
                                                                linewidth=method_styles[method_idx].get('linewidth', 1.5),
                                                                label=label))
                

                ax.grid(True, alpha=0.3)
                ax.set_ylabel(component_labels[component])
                if comp_idx == len(components) - 1:
                    ax.set_xlabel('Time (s)')
            
            # axs[0].legend(handles=legend_elements, loc='upper right', 
            #              bbox_to_anchor=(0.98, 0.98), fontsize='7',
            #              ncol=1)
            # axs[-1].legend(handles=legend_elements, loc='lower left', 
            #   bbox_to_anchor=(0.02, 0.02), fontsize='small',
            #   ncol=1)
            
            axs[0].legend(handles=legend_elements, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=4, mode="expand", borderaxespad=0.)
            
            title = f'{plot_type.capitalize()} - Agent {gt_id}'
            if time_window:
                title += f'\nTime Window: {time_window[0]:.1f}s to {time_window[1]:.1f}s'
            #fig.suptitle(title, y=0.95)
            
            plt.tight_layout()
            
            if output_path:
                time_suffix = f"_t{time_window[0]:.1f}-{time_window[1]:.1f}" if time_window else ""
                save_path = f"{output_path}/comparison_{gt_id}_{plot_type}{time_suffix}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved {plot_type} plot to {save_path}")
            else:
                plt.show()
            plt.close()

def main():
    data_files = [
        "/root/People_tracking/bags/fifth_experiment/results/results_strongsort_pose.csv"
    #    '/root/realsense/case_1/scenario_1/results/results_strongsort_pose.csv'
    ]
    
    assignment_dicts = [
        {'U1': [2], 'U3': [1]}  # StrongSORT+Pose (with ID switch for U1)
    ]
    
    method_names = [
        "YOLOv8-Pose + StrongSORT"
    ]
    
    time_window = (5.0, 45.0)  # None for full duration
    
    # plot_tracking_comparison(
    #     data_files=data_files,
    #     assignment_dicts=assignment_dicts,
    #     method_names=method_names,
    #     time_window=time_window,  # Optional: specify time window
    #     output_path="./figures"
    # )

    plot_tracking_comparison_separated(
        data_files=data_files,
        assignment_dicts=assignment_dicts,
        method_names=method_names,
        time_window=time_window,
        output_path="/root/People_tracking/figures"
        # output_path="/root/realsense/case_1/scenario_1/figures"
    )

if __name__ == '__main__':
    main()