import json
from pathlib import Path
from .visualizer import ComparisonVisualizer
from .metrics import MetricsCollector

def main():
    # Load metrics
    metrics_path = Path("comparison_results/search/metrics.json")
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    # Create metrics collector
    metrics = MetricsCollector(metrics_data)
    
    # Create visualizer
    visualizer = ComparisonVisualizer(metrics)
    
    # Generate all plots
    output_dir = Path("comparison_results/search/visualizations")
    visualizer.plot_all_comparisons(str(output_dir))
    
    print("Visualization complete! Check the comparison_results/search/visualizations directory for the plots.")

if __name__ == "__main__":
    main() 