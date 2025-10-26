"""
Convert fastText format to JSON and extract 1.2M reviews
"""
import bz2
import json

input_file = "data/raw/train.ft.txt.bz2"
output_file = "data/raw/amazon_reviews_1.2M.json"
max_reviews = 1_200_000

print("Converting fastText format to JSON...")
print(f"Input: {input_file}")
print(f"Output: {output_file}")
print(f"Extracting: {max_reviews:,} reviews\n")

count = 0
with bz2.open(input_file, 'rt', encoding='utf-8') as f_in:
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if count >= max_reviews:
                break

            # fastText format: __label__1 text or __label__2 text
            # __label__1 = negative, __label__2 = positive
            line = line.strip()
            if line.startswith('__label__'):
                label = line.split(' ', 1)[0]
                text = line.split(' ', 1)[1] if ' ' in line else ""

                # Convert to rating (1-5 scale)
                rating = 5.0 if label == '__label__2' else 1.0

                review = {
                    'reviewText': text,
                    'overall': rating,
                    'sentiment': 'positive' if rating >= 4 else 'negative'
                }

                json.dump(review, f_out)
                f_out.write('\n')
                count += 1

                if count % 100000 == 0:
                    print(f"  Processed {count:,} reviews...")

print(f"\nDone! Extracted {count:,} reviews to {output_file}")
print("\nNext: Run preprocessing script")
