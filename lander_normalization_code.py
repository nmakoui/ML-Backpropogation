import csv

def normalize_csv(input_path, output_path=None):
    """
    This function reads a CSV file and creates a *normalized* version of the dataset

    HOW NORMALISATION WORKS:
    We use classic *min–max normalization* for each column:

        normalized_value = (value - min(column)) / (max(column) - min(column))

    This maps every feature column into the range [0,1], which is ideal for MATLAB
    analysis and for comparing hyperparameter behaviours.
    """

    # -------------------------------------------------------
    # 1. READ THE CSV FILE INTO MEMORY
    # -------------------------------------------------------
    # csv.reader allows us to work with CSV files using only built-in Python tools.
    with open(input_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Optional: assume first row is header (column names).
    # If your dataset does NOT have headers, remove these two lines.
    header = rows[0]
    data_rows = rows[1:]

    # -------------------------------------------------------
    # 2. CONVERT ALL VALUES TO FLOAT
    # -------------------------------------------------------
    # We convert all string values into floats so we can apply mathematical operations.
    data = []
    for row in data_rows:
        # Convert every value in the row to float
        float_row = [float(x) for x in row]
        data.append(float_row)

    # -------------------------------------------------------
    # 3. COMPUTE MIN AND MAX OF EACH COLUMN
    # -------------------------------------------------------
    # This step finds the smallest and largest values in each column,
    # which we need for min–max normalization.
    num_cols = len(data[0])

    mins = []
    maxs = []

    for col in range(num_cols):
        col_values = [row[col] for row in data]
        mins.append(min(col_values))
        maxs.append(max(col_values))

    # Prevent division by zero if a column has no variation.
    ranges = []
    for i in range(num_cols):
        rng = maxs[i] - mins[i]
        if rng == 0:
            rng = 1e-12  # tiny number to avoid crashing
        ranges.append(rng)

    # -------------------------------------------------------
    # 4. APPLY MIN–MAX NORMALIZATION TO EVERY VALUE
    # -------------------------------------------------------
    normalized_rows = []
    for row in data:
        norm_row = []
        for i in range(num_cols):
            norm_value = (row[i] - mins[i]) / ranges[i]
            norm_row.append(norm_value)
        normalized_rows.append(norm_row)

    # -------------------------------------------------------
    # 5. DETERMINE THE OUTPUT FILE NAME
    # -------------------------------------------------------
    if output_path is None:
        # Append "_normalized" before .csv
        output_path = input_path.replace(".csv", "_normalized.csv")

    # -------------------------------------------------------
    # 6. SAVE THE NORMALISED DATA BACK TO A CSV FILE
    # -------------------------------------------------------
    # We write both the header (optional) and the normalized rows.
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(normalized_rows)

    print(f"✔ Normalized file saved as: {output_path}")
    return output_path


# -------------------------------------------------------
# RUNNING THIS SCRIPT DIRECTLY (OPTIONAL)
# -------------------------------------------------------
# If this file is run directly (not imported), we automatically normalize the main dataset.
if __name__ == "__main__":
    input_file = "ce889_dataCollection.csv"
    normalize_csv(input_file)
