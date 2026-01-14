"""
Visualization utilities for supply chain network and results.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from ..data.models import SupplyChainNetwork
from ..hybrid.coordinator import HybridSolverResult


class NetworkVisualizer:
    """Visualize supply chain network and optimization results."""
    
    def __init__(self, network: SupplyChainNetwork):
        """
        Initialize visualizer.
        
        Args:
            network: Supply chain network
        """
        self.network = network
    
    def plot_network(
        self,
        x_assign: Optional[Dict[Tuple[int, int], int]] = None,
        y_assign: Optional[Dict[Tuple[int, int], int]] = None,
        title: str = "Supply Chain Network",
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the supply chain network with assignments.
        
        Args:
            x_assign: Supplier-warehouse assignments
            y_assign: Warehouse-customer assignments
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add supplier nodes
        for supplier in self.network.suppliers:
            G.add_node(
                f"S{supplier.id}",
                pos=supplier.location,
                node_type="supplier"
            )
        
        # Add warehouse nodes
        for warehouse in self.network.warehouses:
            G.add_node(
                f"W{warehouse.id}",
                pos=warehouse.location,
                node_type="warehouse"
            )
        
        # Add customer nodes
        for customer in self.network.customers:
            node_type = "factory" if customer.is_factory else f"customer_tier{customer.tier}"
            G.add_node(
                f"C{customer.id}",
                pos=customer.location,
                node_type=node_type
            )
        
        # Add edges based on assignments
        if x_assign:
            for (i, j), v in x_assign.items():
                if v == 1:
                    G.add_edge(f"S{i}", f"W{j}", edge_type="sw")
        
        if y_assign:
            for (j, k), v in y_assign.items():
                if v == 1:
                    G.add_edge(f"W{j}", f"C{k}", edge_type="wc")
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes by type
        node_colors = {
            "supplier": "#3498db",      # Blue
            "warehouse": "#f39c12",     # Orange
            "factory": "#e74c3c",       # Red
            "customer_tier1": "#27ae60", # Green
            "customer_tier2": "#9b59b6", # Purple
            "customer_tier3": "#95a5a6"  # Gray
        }
        
        for node_type, color in node_colors.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
            if nodes:
                sizes = 800 if node_type in ["factory", "warehouse"] else 400
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes,
                    node_color=color, node_size=sizes, ax=ax
                )
        
        # Draw edges
        sw_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'sw']
        wc_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'wc']
        
        if sw_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=sw_edges,
                edge_color='#3498db', alpha=0.6, arrows=True,
                arrowsize=15, ax=ax
            )
        
        if wc_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=wc_edges,
                edge_color='#27ae60', alpha=0.6, arrows=True,
                arrowsize=15, ax=ax
            )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='#3498db', label='Suppliers'),
            mpatches.Patch(color='#f39c12', label='Warehouses'),
            mpatches.Patch(color='#e74c3c', label='Factory'),
            mpatches.Patch(color='#27ae60', label='Tier 1 Customers'),
            mpatches.Patch(color='#9b59b6', label='Tier 2 Customers'),
            mpatches.Patch(color='#95a5a6', label='Tier 3 Customers'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_cost_breakdown(
        self,
        result: HybridSolverResult,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cost breakdown across scenarios.
        
        Args:
            result: Hybrid solver result
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Cost breakdown pie chart
        ax1 = axes[0]
        costs = [result.stage1_value, result.expected_cost]
        labels = ['Stage 1\n(Assignment)', 'Stage 2\n(Operations)']
        colors = ['#3498db', '#e74c3c']
        
        ax1.pie(costs, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0))
        ax1.set_title('Cost Breakdown', fontweight='bold')
        
        # Right: Scenario costs bar chart
        ax2 = axes[1]
        scenarios = list(result.scenario_costs.keys())
        costs = list(result.scenario_costs.values())
        
        bars = ax2.bar(scenarios, costs, color='#27ae60', alpha=0.8)
        ax2.axhline(y=result.expected_cost, color='#e74c3c', linestyle='--',
                   label=f'Expected: {result.expected_cost:.0f}')
        ax2.axhline(y=result.cost_statistics.get('cvar_95', 0),
                   color='#f39c12', linestyle=':',
                   label=f'CVaR 95%: {result.cost_statistics.get("cvar_95", 0):.0f}')
        
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Cost')
        ax2.set_title('Scenario Costs', fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def plot_network(
    network: SupplyChainNetwork,
    x_assign: Optional[Dict] = None,
    y_assign: Optional[Dict] = None,
    **kwargs
) -> plt.Figure:
    """Convenience function to plot network."""
    viz = NetworkVisualizer(network)
    return viz.plot_network(x_assign, y_assign, **kwargs)


def plot_results(
    network: SupplyChainNetwork,
    result: HybridSolverResult,
    save_dir: Optional[str] = None
) -> List[plt.Figure]:
    """
    Plot all result visualizations.
    
    Args:
        network: Supply chain network
        result: Hybrid solver result
        save_dir: Directory to save figures
        
    Returns:
        List of figures
    """
    viz = NetworkVisualizer(network)
    figures = []
    
    # Network plot
    fig1 = viz.plot_network(
        result.x_assign, result.y_assign,
        title="Optimized Supply Chain Network",
        save_path=f"{save_dir}/network.png" if save_dir else None
    )
    figures.append(fig1)
    
    # Cost breakdown
    fig2 = viz.plot_cost_breakdown(
        result,
        save_path=f"{save_dir}/cost_breakdown.png" if save_dir else None
    )
    figures.append(fig2)
    
    return figures
