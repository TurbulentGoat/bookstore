import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import seaborn as sns
from math import comb


def simulate_book_collection(total_books, books_per_shop, num_simulations):
    """
    Simulates collecting unique books by visiting multiple shops.
    Returns both the number of shops visited and the collection progress for each simulation.
    """
    results = []
    collection_progress = []

    for sim in range(num_simulations):
        unique_books = set()
        shops_visited = 0
        progress = []

        while len(unique_books) < total_books:
            shops_visited += 1
            # Generate a random set of books from the current shop
            books_bought = set(random.sample(range(total_books), min(books_per_shop, total_books)))
            # Add new unique books to the collection
            unique_books.update(books_bought)
            # Record progress after each shop visit
            progress.append((shops_visited, len(unique_books)))

        results.append(shops_visited)
        collection_progress.append(progress)

    return results, collection_progress


def calculate_probabilities(results, max_shops):
    """
    Calculates probability of completing collection within different numbers of shop visits.
    """
    probabilities = []
    for shops in range(1, max_shops + 1):
        prob = sum(1 for x in results if x <= shops) / len(results)
        probabilities.append((shops, prob))
    return probabilities


def create_visualizations(results, collection_progress, probabilities, total_books):
    """
    Creates an enhanced visualization with multiple subplots showing different aspects of the data.
    """
    # Set up the figure with a grid layout
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig)
    plt.rcParams['axes.titlesize'] = 12  # sets default title size for all plots

    # 1. Distribution plot with normal curve and probability annotations
    ax1 = fig.add_subplot(gs[0, :])
    counts, bins, _ = ax1.hist(results, bins=30, density=True, alpha=0.7,
                              color='skyblue', label='Simulation Results')

    # Fit and plot normal distribution (optional)
    # mu, sigma = stats.norm.fit(results)
    # x = np.linspace(min(bins), max(bins), 100)
    # best_fit = stats.norm.pdf(x, mu, sigma)
    # ax1.plot(x, best_fit, 'r-', linewidth=2,
    #          label=f'Normal Distribution\nμ={mu:.1f}, σ={sigma:.1f}')

    ax1.set_title('Distribution of Shops Visited to Complete Collection')
    ax1.set_xlabel('Number of Shops Visited')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add probability annotations
    key_probs = [0.25, 0.5, 0.75, 0.9, 1]
    for target_prob in key_probs:
        # Find the first shop where cumulative probability >= target_prob
        shop = next((s for s, p in probabilities if p >= target_prob), None)
        if shop is not None:
            ax1.axvline(x=shop, color='green', linestyle='--', alpha=0.7)
            ax1.text(shop, ax1.get_ylim()[1]*0.9, f'{target_prob*100:.0f}%\n({shop} shops)',
                     rotation=90, ha='right', va='center', backgroundcolor='white')

    # 2. Collection Progress Plot
    ax2 = fig.add_subplot(gs[1, 0])
    # Plot progress lines for a random sample of simulations
    sample_size = min(50, len(collection_progress))
    random_samples = random.sample(collection_progress, sample_size)

    for progress in random_samples:
        shops, books = zip(*progress)
        ax2.plot(shops, books, alpha=0.1, color='blue')

    # Plot average progress line
    avg_progress = []
    max_shops = max(len(p) for p in collection_progress)
    for shop in range(1, max_shops + 1):
        books_at_shop = [p[shop-1][1] for p in collection_progress if len(p) >= shop]
        if books_at_shop:
            avg_progress.append(np.mean(books_at_shop))

    ax2.plot(range(1, len(avg_progress) + 1), avg_progress,
             color='red', linewidth=2, label='Average Progress')

    ax2.set_title('Collection Progress Over Time')
    ax2.set_xlabel('Shops Visited')
    ax2.set_ylabel('Books Collected')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Probability Line Plot instead of Heat Map
    ax3 = fig.add_subplot(gs[1, 1])
    shops = [p[0] for p in probabilities]
    probs = [p[1] for p in probabilities]

    # Create line plot
    ax3.plot(shops, probs, 'b-', linewidth=2)
    ax3.fill_between(shops, probs, alpha=0.3, color='blue')

    # Add grid for better readability
    ax3.grid(True, alpha=0.3)

    # Add probability markers at key thresholds
    for target_prob in key_probs:
        # Find the first shop where cumulative probability >= target_prob
        shop = next((s for s, p in probabilities if p >= target_prob), None)
        if shop is not None:
            prob = next(p for s, p in probabilities if s == shop)
            ax3.plot(shop, prob, 'ro')
            ax3.annotate(f'{prob:.0%}', 
                        (shop, prob),
                        xytext=(10, 0),
                        textcoords='offset points',
                        ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Improve axis labels and formatting
    ax3.set_title('Probability of Completing Collection vs Shops Visited')
    ax3.set_xlabel('Number of Shops Visited')
    ax3.set_ylabel('Probability of Completion')
    ax3.set_ylim(0, 1.05)  # Set y-axis from 0 to 1.05 for better visualization

    # Format y-axis as percentages
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add horizontal lines at key probability levels
    for prob in key_probs:
        ax3.axhline(y=prob, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def prob_j_new_books(n, k, i, j):
    """
    n: total unique books
    k: books selected per shop
    i: books already collected
    j: new books we're trying to get
    """
    # Number of ways to choose j new books from (n-i) unseen books
    ways_new = comb(n - i, j)
    # Number of ways to choose (k-j) already-seen books from i books
    ways_old = comb(i, k - j)
    # Total number of ways to choose k books from n books
    total_ways = comb(n, k)

    return (ways_new * ways_old) / total_ways if total_ways > 0 else 0


def expected_shops(n, k):
    """
    Calculate expected number of shops needed
    n: total unique books
    k: books selected per shop
    """
    total = 0.0
    for i in range(n):
        # Probability of getting at least one new book
        p_at_least_one = 1 - prob_j_new_books(n, k, i, 0)
        if p_at_least_one > 0:
            total += 1 / p_at_least_one
    return total


def print_statistics(results, probabilities):
    """
    Prints enhanced statistics about the simulation results.
    """
    mean_shops = np.mean(results)
    median_shops = np.median(results)
    std_dev = np.std(results)
    
    print("\nSimulation Statistics:")
    print(f"Average shops needed: {mean_shops:.1f}")
    print(f"Median shops needed: {median_shops:.1f}")
    print(f"Standard deviation: {std_dev:.1f}")
    print(f"Minimum shops needed: {min(results)}")
    print(f"Maximum shops needed: {max(results)}")
    
    print("\nProbability of completing collection:")
    key_probs = [0.25, 0.5, 0.75, 0.9, 0.95, 1]
    for target_prob in key_probs:
        shop = next((s for s, p in probabilities if p >= target_prob), None)
        if shop is not None:
            prob = next(p for s, p in probabilities if s == shop)
            print(f"{shop} shops gives a {prob:.1%} chance of completion")
        else:
            print(f"Could not determine shop count for {target_prob*100:.0f}% probability")


def main():
    print("Welcome to the Enhanced Book Collection Simulator!")
    print("\nThis simulation will help you understand:")
    print("- How many shops you'll likely need to visit")
    print("- The probability of completing your collection in X visits")
    print("- How your collection typically grows over time")
    
    # Input validation
    while True:
        try:
            total_books = int(input("\nHow many total unique books do you need to collect? "))
            if total_books <= 0:
                print("Please enter a positive integer for the total number of books.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
    
    while True:
        try:
            books_per_shop = int(input("How many books can you buy from each shop? "))
            if books_per_shop <= 0 or books_per_shop > total_books:
                print(f"Please enter a positive integer up to {total_books} for books per shop.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
    
    while True:
        try:
            num_simulations = int(input("How many times should we run the simulation? "))
            if num_simulations <= 0:
                print("Please enter a positive integer for the number of simulations.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
    
    print(f"\nRunning {num_simulations} simulations...")
    results, collection_progress = simulate_book_collection(total_books, books_per_shop, num_simulations)
    
    # Calculate probabilities up to the maximum number of shops needed
    max_shops = max(results)
    probabilities = calculate_probabilities(results, max_shops)
    
    # Print statistics
    print_statistics(results, probabilities)
    
    # Calculate and print expected number of shops
    expected = expected_shops(total_books, books_per_shop)
    print(f"\nTheoretical expected number of shops needed: {expected:.2f}")
    
    # Create visualizations
    create_visualizations(results, collection_progress, probabilities, total_books)


if __name__ == "__main__":
    main()

