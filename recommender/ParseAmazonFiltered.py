#!/usr/bin/env python3

import sys
import json
from collections import defaultdict

def parse_and_filter_amazon_data(filepath, min_ratings=2, max_ratings=100):

    print(f"Parsing {filepath}...")
    
    # Collect all customer ratings first
    customer_ratings = defaultdict(dict)  # customer -> {asin: rating}
    asin_metadata = {}
    
    current_asin = None
    current_group = None
    current_title = None
    current_salesrank = None
    current_avg_rating = None
    current_categories = []
    in_categories = False
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('ASIN:'):
                current_asin = line.split('ASIN:')[1].strip()
                current_group = None
                current_title = None
                current_salesrank = None
                current_avg_rating = None
                current_categories = []
                in_categories = False
            
            elif line.startswith('title:'):
                current_title = line.split('title:')[1].strip()
            
            elif line.startswith('group:'):
                current_group = line.split('group:')[1].strip()
            
            elif line.startswith('reviews:'):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'rating:':
                        if i + 1 < len(parts):
                            try:
                                current_avg_rating = float(parts[i + 1])
                            except ValueError:
                                pass
                
                # Store metadata for Music items
                if current_group == 'Music' and current_asin:
                    asin_metadata[current_asin] = {
                        'title': current_title,
                        'avg_rating': current_avg_rating
                    }
            
            elif current_group == 'Music' and 'cutomer:' in line and 'rating:' in line:
                parts = line.split()
                
                customer_id = None
                rating = None
                
                for i, part in enumerate(parts):
                    if part == 'cutomer:':
                        if i + 1 < len(parts):
                            customer_id = parts[i + 1]
                    elif part == 'rating:':
                        if i + 1 < len(parts):
                            try:
                                rating = int(parts[i + 1])
                            except ValueError:
                                pass
                
                if customer_id and rating:
                    customer_ratings[customer_id][current_asin] = rating
    
    # Filter customers based on rating counts
    print(f"\nFiltering customers (min={min_ratings}, max={max_ratings})...")
    
    customer_rating_counts = {cust_id: len(ratings) for cust_id, ratings in customer_ratings.items()}
    
    valid_customers = {
        cust_id for cust_id, count in customer_rating_counts.items()
        if min_ratings <= count <= max_ratings
    }
    
    # Filter the ratings dict
    filtered_customer_ratings = {
        cust_id: customer_ratings[cust_id] 
        for cust_id in valid_customers
    }
    
    # Statistics
    total_customers = len(customer_rating_counts)
    removed_too_few = sum(1 for count in customer_rating_counts.values() if count < min_ratings)
    removed_too_many = sum(1 for count in customer_rating_counts.values() if count > max_ratings)
    kept_customers = len(valid_customers)
    
    total_ratings_before = sum(customer_rating_counts.values())
    total_ratings_after = sum(len(filtered_customer_ratings[cust_id]) for cust_id in valid_customers)
    
    print(f"\n=== Filtering Statistics ===")
    print(f"Customers:")
    print(f"  Total: {total_customers}")
    print(f"  Removed (< {min_ratings} ratings): {removed_too_few} ({100*removed_too_few/total_customers:.1f}%)")
    print(f"  Removed (> {max_ratings} ratings): {removed_too_many} ({100*removed_too_many/total_customers:.1f}%)")
    print(f"  Kept: {kept_customers} ({100*kept_customers/total_customers:.1f}%)")
    
    print(f"\nRatings:")
    print(f"  Total before: {total_ratings_before}")
    print(f"  Total after: {total_ratings_after} ({100*total_ratings_after/total_ratings_before:.1f}%)")
    
    # Distribution statistics
    kept_counts = [customer_rating_counts[cust_id] for cust_id in valid_customers]
    if kept_counts:
        kept_counts_sorted = sorted(kept_counts)
        print(f"\nRating count distribution (kept customers):")
        print(f"  Min: {min(kept_counts)}")
        print(f"  Q1 (25%): {kept_counts_sorted[len(kept_counts)//4]}")
        print(f"  Median: {kept_counts_sorted[len(kept_counts)//2]}")
        print(f"  Q3 (75%): {kept_counts_sorted[3*len(kept_counts)//4]}")
        print(f"  Max: {max(kept_counts)}")
        print(f"  Average: {sum(kept_counts)/len(kept_counts):.1f}")
    
    # Get unique ASINs from filtered ratings
    unique_asins = set()
    for ratings in filtered_customer_ratings.values():
        unique_asins.update(ratings.keys())
    
    # Filter metadata to only include ASINs that appear in filtered ratings
    filtered_metadata = {
        asin: meta for asin, meta in asin_metadata.items()
        if asin in unique_asins
    }
    
    print(f"\nASINs:")
    print(f"  Total in original data: {len(asin_metadata)}")
    print(f"  In filtered ratings: {len(filtered_metadata)}")
    
    return filtered_customer_ratings, filtered_metadata, customer_rating_counts, valid_customers


def write_customer_ratings(customer_ratings, output_file):
    """Write customer ratings in TSV format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CustomerID\tASIN\tRating\n")
        for customer_id in sorted(customer_ratings.keys()):
            for asin, rating in sorted(customer_ratings[customer_id].items()):
                f.write(f"{customer_id}\t{asin}\t{rating}\n")


def write_asin_metadata(asin_metadata, output_file):
    """Write ASIN metadata in TSV format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ASIN\tTitle\tAvgRating\n")
        for asin in sorted(asin_metadata.keys()):
            meta = asin_metadata[asin]
            f.write(f"{asin}\t{meta.get('title', '')}\t"
                f"{meta.get('avg_rating', '')}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_amazon_filtered.py <input_file> [min_ratings] [max_ratings]")
        print()
        print("Default: min_ratings=2, max_ratings=100")
        print()
        print("Outputs:")
        print("  - customer_ratings_filtered.txt: All ratings (TSV format)")
        print("  - asin_metadata_filtered.txt: ASIN info (TSV)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    min_ratings = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_ratings = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Parse and filter
    customer_ratings, asin_metadata, all_counts, valid_customers = \
        parse_and_filter_amazon_data(input_file, min_ratings, max_ratings)
    
    # Write outputs
    print(f"\nWriting output files...")
    
    write_customer_ratings(customer_ratings, 'customer_ratings_filtered.txt')
    print("  ✓ customer_ratings_filtered.txt")
    
    write_asin_metadata(asin_metadata, 'asin_metadata_filtered.txt')
    print("  ✓ asin_metadata_filtered.txt")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
