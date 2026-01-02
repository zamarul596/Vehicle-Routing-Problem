# GENETIC ALGORITHM FOR VEHICLE ROUTING PROBLEM (VRP)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict



# RANDOM STATE MANAGEMENT

class RandomState:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def shuffle(self, x):
        self.rng.shuffle(x)
    
    def choice(self, a, size=None, replace=True):
        return self.rng.choice(a, size=size, replace=replace)
    
    def random(self):
        return self.rng.random()

# Global random state instance
rng: RandomState = None

def set_random_seed(seed: int):
    global rng
    rng = RandomState(seed)



# DATA LOADING

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def extract_problem_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    coords = df[['x', 'y']].values
    demands = df['demand'].values
    capacity = df['vehicle_capacity'].iloc[0]
    return coords, demands, capacity



# DISTANCE CALCULATIONS

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt(
                (coords[i, 0] - coords[j, 0])**2 + 
                (coords[i, 1] - coords[j, 1])**2
            )
    return dist_matrix


def calculate_route_distance(route: List[int], dist_matrix: np.ndarray) -> float:
    if not route:
        return 0.0
    distance = dist_matrix[0, route[0]]  # Depot to first customer
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i + 1]]
    distance += dist_matrix[route[-1], 0]  # Last customer to depot
    return distance


def calculate_total_distance(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    return sum(calculate_route_distance(route, dist_matrix) for route in routes)



# SOLUTION ENCODING/DECODING

def decode_chromosome(chromosome: List[int], demands: np.ndarray, capacity: int) -> List[List[int]]:
    routes = []
    current_route = []
    current_load = 0
    
    for customer in chromosome:
        demand = demands[customer]
        
        if current_load + demand > capacity:
            if current_route:
                routes.append(current_route)
            current_route = []
            current_load = 0
        
        current_route.append(customer)
        current_load += demand
    
    if current_route:
        routes.append(current_route)
    
    return routes



# GENETIC ALGORITHM OPERATORS

def create_initial_population(pop_size: int, num_customers: int) -> List[List[int]]:
    global rng
    population = []
    customers = list(range(1, num_customers + 1))  # Exclude depot (0)
    for _ in range(pop_size):
        chromosome = customers.copy()
        rng.shuffle(chromosome)
        population.append(chromosome)
    return population


def evaluate_fitness(chromosome: List[int], demands: np.ndarray, capacity: int, dist_matrix: np.ndarray) -> float:
    
    routes = decode_chromosome(chromosome, demands, capacity)
    total_dist = calculate_total_distance(routes, dist_matrix)
    return 1.0 / total_dist if total_dist > 0 else 0


def tournament_selection(population: List[List[int]], fitness_scores: List[float], tournament_size: int = 3) -> List[int]:
    global rng
    indices = rng.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
    return population[best_idx].copy()


def order_crossover_single(parent1: List[int], parent2: List[int]) -> List[int]:
    global rng
    size = len(parent1)
    
    # Select crossover points
    point1, point2 = sorted(rng.choice(size, 2, replace=False))
    
    # Create child
    child = [None] * size
    
    # Copy segment from parent1
    child[point1:point2] = parent1[point1:point2]
    
    # Fill remaining positions with order from parent2
    current_pos = point2
    for gene in parent2[point2:] + parent2[:point2]:
        if gene not in child:
            if current_pos >= size:
                current_pos = 0
            while child[current_pos] is not None:
                current_pos += 1
                if current_pos >= size:
                    current_pos = 0
            child[current_pos] = gene
    
    return child


def swap_mutation(chromosome: List[int]) -> List[int]:
    global rng
    mutated = chromosome.copy()
    i, j = rng.choice(len(mutated), 2, replace=False)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def inversion_mutation(chromosome: List[int]) -> List[int]:
    global rng
    mutated = chromosome.copy()
    idx1, idx2 = sorted(rng.choice(len(mutated), 2, replace=False))
    mutated[idx1:idx2] = mutated[idx1:idx2][::-1]
    return mutated



# GENETIC ALGORITHM MAIN LOOP

def run_genetic_algorithm(
    coords: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    pop_size: int,
    crossover_rate: float,
    mutation_rate: float,
    generations: int,
    seed: int = 42,
    progress_callback=None
) -> Dict:

    global rng
    
    start_time = time.time()
    
    set_random_seed(seed)
    
    # Initialize
    num_customers = len(coords) - 1  # Exclude depot
    dist_matrix = calculate_distance_matrix(coords)
    population = create_initial_population(pop_size, num_customers)
    
    # Track convergence
    best_distances = []
    best_chromosome = None
    best_fitness = -float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [
            evaluate_fitness(chrom, demands, capacity, dist_matrix) 
            for chrom in population
        ]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_chromosome = population[max_fitness_idx].copy()
        
        # Record best distance for convergence plot
        best_distance = 1.0 / best_fitness if best_fitness > 0 else float('inf')
        best_distances.append(best_distance)
        
        # Create new population
        new_population = []
        
        # Elitism - keep best individual
        new_population.append(best_chromosome.copy())
        
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # Crossover
            if rng.random() < crossover_rate:
                child1 = order_crossover_single(parent1, parent2)
                child2 = order_crossover_single(parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                child1 = swap_mutation(child1) if rng.random() < 0.5 else inversion_mutation(child1)
            if rng.random() < mutation_rate:
                child2 = swap_mutation(child2) if rng.random() < 0.5 else inversion_mutation(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
        
        if progress_callback:
            progress_callback((gen + 1) / generations)
    
    runtime = time.time() - start_time
    
    # Decode best solution
    best_routes = decode_chromosome(best_chromosome, demands, capacity)
    final_distance = calculate_total_distance(best_routes, dist_matrix)
    
    return {
        'best_chromosome': best_chromosome,
        'best_routes': best_routes,
        'best_distance': final_distance,
        'num_routes': len(best_routes),
        'convergence': best_distances,
        'runtime': runtime,
        'dist_matrix': dist_matrix
    }



# VISUALIZATION FUNCTIONS

def plot_convergence(baseline_results: Dict, custom_results: Dict = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot baseline
    ax.plot(baseline_results['convergence'], 'b-', linewidth=2, 
            label=f"Baseline (Final: {baseline_results['best_distance']:.4f})")
    
    # Plot custom if user run new parameters
    if custom_results:
        ax.plot(custom_results['convergence'], 'r--', linewidth=2,
                label=f"Custom (Final: {custom_results['best_distance']:.4f})")
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Distance', fontsize=12)
    ax.set_title('GA Convergence Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_routes(coords: np.ndarray, routes: List[List[int]], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette for routes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 10)))
    
    # Plot depot
    ax.scatter(coords[0, 0], coords[0, 1], c='red', s=200, marker='s', 
               zorder=5, label='Depot', edgecolors='black', linewidth=2)
    
    # Plot customers
    for i in range(1, len(coords)):
        ax.scatter(coords[i, 0], coords[i, 1], c='blue', s=100, marker='o',
                   zorder=4, edgecolors='black', linewidth=1)
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Plot routes
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        
        # Depot to first customer
        ax.plot([coords[0, 0], coords[route[0], 0]], 
                [coords[0, 1], coords[route[0], 1]], 
                c=color, linewidth=2, alpha=0.7)
        
        # Between customers
        for i in range(len(route) - 1):
            ax.plot([coords[route[i], 0], coords[route[i + 1], 0]],
                    [coords[route[i], 1], coords[route[i + 1], 1]],
                    c=color, linewidth=2, alpha=0.7)
        
        # Last customer to depot
        ax.plot([coords[route[-1], 0], coords[0, 0]],
                [coords[route[-1], 1], coords[0, 1]],
                c=color, linewidth=2, alpha=0.7, label=f'Route {idx + 1}')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig



# STREAMLIT APPLICATION


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="VRP Solver - Genetic Algorithm",
        layout="wide"
    )
    
    # Title and description
    st.title("Vehicle Routing Problem Solver")
    st.markdown("""
    ### Genetic Algorithm with Comparative Analysis
    
    This dashboard solves the **Vehicle Routing Problem (VRP)** using a **Genetic Algorithm**.
    Compare the baseline configuration with your custom parameters to analyze performance.
    """)
    
    st.divider()
    
    # Load data
    try:
        df = load_data("vrp_raw_dataset.csv")
        coords, demands, capacity = extract_problem_data(df)
        st.success(f"Dataset loaded: {len(coords) - 1} customers, Vehicle Capacity: {capacity}")
    except FileNotFoundError:
        st.error("Dataset file 'vrp_raw_dataset.csv' not found!")
        return
    
    # ========================================================================
    # SIDEBAR - Parameters
    # ========================================================================
    
    st.sidebar.header("GA Parameters")
    
    # Baseline configuration (fixed)
    st.sidebar.subheader("Baseline Configuration (Fixed)")
    st.sidebar.info("""
    - Population Size: **200**
    - Crossover Rate: **0.6**
    - Mutation Rate: **0.05**
    - Generations: **100**
    """)
    
    st.sidebar.divider()
    
    # Custom parameters
    st.sidebar.subheader("Custom Configuration")
    
    custom_pop_size = st.sidebar.slider(
        "Population Size", 
        min_value=50, max_value=300, value=100, step=10,
        help="Number of individuals in the population"
    )
    
    custom_crossover_rate = st.sidebar.slider(
        "Crossover Rate", 
        min_value=0.6, max_value=0.9, value=0.8, step=0.05,
        help="Probability of crossover between parents"
    )
    
    custom_mutation_rate = st.sidebar.slider(
        "Mutation Rate", 
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Probability of mutation in offspring"
    )
    
    custom_generations = st.sidebar.slider(
        "Generations", 
        min_value=50, max_value=500, value=100, step=10,
        help="Number of generations to evolve"
    )
    
    st.sidebar.divider()
    
    # Random seed for reproducibility
    seed = st.sidebar.number_input(
        "Random Seed", 
        min_value=0, max_value=9999, value=42,
        help="Set seed for reproducible results (Standard use is 42)"
    )
    
    run_custom = st.sidebar.button("Run Custom GA", type="primary", use_container_width=True)
    


    # MAIN CONTENT

    # Initialize session state
    if 'baseline_results' not in st.session_state:
        st.session_state.baseline_results = None
    if 'custom_results' not in st.session_state:
        st.session_state.custom_results = None
    
    # Run baseline if not already done
    if st.session_state.baseline_results is None:
        st.subheader("Running Baseline GA...")
        
        progress_bar = st.progress(0)
        
        def update_progress(progress):
            progress_bar.progress(progress)
        
        st.session_state.baseline_results = run_genetic_algorithm(
            coords, demands, capacity,
            pop_size=200,
            crossover_rate=0.6,
            mutation_rate=0.05,
            generations=100,
            seed=42,
            progress_callback=update_progress
        )
        
        progress_bar.empty()
        st.rerun()
    
    # Run custom GA if button clicked
    if run_custom:
        st.subheader("Running Custom GA...")
        
        progress_bar = st.progress(0)
        
        def update_progress(progress):
            progress_bar.progress(progress)
        
        st.session_state.custom_results = run_genetic_algorithm(
            coords, demands, capacity,
            pop_size=custom_pop_size,
            crossover_rate=custom_crossover_rate,
            mutation_rate=custom_mutation_rate,
            generations=custom_generations,
            seed=seed,
            progress_callback=update_progress
        )
        
        progress_bar.empty()
        st.rerun()
    
  

    # RESULTS DISPLAY
    
    baseline = st.session_state.baseline_results
    custom = st.session_state.custom_results
    
    # Performance Comparison Table
    st.header("Performance Comparison")
    
    comparison_data = {
        "Metric": ["Best Total Distance", "Number of Routes (Vehicles)", "Runtime (seconds)"],
        "Baseline": [
            f"{baseline['best_distance']:.4f}",
            baseline['num_routes'],
            f"{baseline['runtime']:.2f}"
        ]
    }
    
    if custom:
        comparison_data["Custom"] = [
            f"{custom['best_distance']:.4f}",
            custom['num_routes'],
            f"{custom['runtime']:.2f}"
        ]
        
        # Calculate improvement
        dist_diff = ((custom['best_distance'] - baseline['best_distance']) / baseline['best_distance']) * 100
        comparison_data["Difference"] = [
            f"{dist_diff:+.2f}%",
            f"{custom['num_routes'] - baseline['num_routes']:+d}",
            f"{custom['runtime'] - baseline['runtime']:+.2f}s"
        ]
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Detailed parameters comparison
    if custom:
        st.subheader("Parameter Configuration Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Baseline Parameters:**")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | Population Size | 200 |
            | Crossover Rate | 0.6 |
            | Mutation Rate | 0.05 |
            | Generations | 100 |
            """)
        
        with col2:
            st.markdown("**Custom Parameters:**")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Population Size | {custom_pop_size} |
            | Crossover Rate | {custom_crossover_rate} |
            | Mutation Rate | {custom_mutation_rate} |
            | Generations | {custom_generations} |
            """)
    
    st.divider()
    
    # Convergence Plot
    st.header("Convergence Analysis")
    
    fig_convergence = plot_convergence(baseline, custom)
    st.pyplot(fig_convergence)
    
    if custom:
        # Convergence analysis insights
        baseline_final = baseline['convergence'][-1]
        custom_final = custom['convergence'][-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Baseline Final Distance", 
                f"{baseline_final:.4f}"
            )
        with col2:
            st.metric(
                "Custom Final Distance", 
                f"{custom_final:.4f}",
                delta=f"{custom_final - baseline_final:.4f}"
            )
        with col3:
            improvement = "Better" if custom_final < baseline_final else "Worse"
            st.metric(
                "Performance",
                improvement,
                delta=f"{abs(custom_final - baseline_final) / baseline_final * 100:.2f}%"
            )
    
    st.divider()
    
    # Route Visualization
    st.header("Route Visualization")
    
    if custom:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseline Solution")
            fig_baseline = plot_routes(
                coords, baseline['best_routes'],
                f"Baseline Routes (Distance: {baseline['best_distance']:.4f})"
            )
            st.pyplot(fig_baseline)
            
            # Route details
            with st.expander("Baseline Route Details"):
                for i, route in enumerate(baseline['best_routes']):
                    route_demand = sum(demands[c] for c in route)
                    route_dist = calculate_route_distance(route, baseline['dist_matrix'])
                    st.write(f"**Route {i+1}:** Depot → {' → '.join(map(str, route))} → Depot")
                    st.write(f"   Load: {route_demand}/{capacity}, Distance: {route_dist:.4f}")
        
        with col2:
            st.subheader("Custom Solution")
            fig_custom = plot_routes(
                coords, custom['best_routes'],
                f"Custom Routes (Distance: {custom['best_distance']:.4f})"
            )
            st.pyplot(fig_custom)
            
            # Route details
            with st.expander("Custom Route Details"):
                for i, route in enumerate(custom['best_routes']):
                    route_demand = sum(demands[c] for c in route)
                    route_dist = calculate_route_distance(route, custom['dist_matrix'])
                    st.write(f"**Route {i+1}:** Depot → {' → '.join(map(str, route))} → Depot")
                    st.write(f"   Load: {route_demand}/{capacity}, Distance: {route_dist:.4f}")
    else:
        st.subheader("Baseline Solution")
        fig_baseline = plot_routes(
            coords, baseline['best_routes'],
            f"Baseline Routes (Distance: {baseline['best_distance']:.4f})"
        )
        st.pyplot(fig_baseline)
        
        # Route details
        with st.expander("Baseline Route Details"):
            for i, route in enumerate(baseline['best_routes']):
                route_demand = sum(demands[c] for c in route)
                route_dist = calculate_route_distance(route, baseline['dist_matrix'])
                st.write(f"**Route {i+1}:** Depot → {' → '.join(map(str, route))} → Depot")
                st.write(f"   Load: {route_demand}/{capacity}, Distance: {route_dist:.4f}")
        
        st.info("You can use the sidebar to configure custom parameters and click 'Run Custom GA' to compare solutions.")
    
    st.divider()
    
    # Dataset Overview
    st.header("Dataset Overview")
    
    with st.expander("View Dataset"):
        st.dataframe(df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(coords) - 1)
    with col2:
        st.metric("Total Demand", int(demands.sum()))
    with col3:
        st.metric("Vehicle Capacity", capacity)
    
