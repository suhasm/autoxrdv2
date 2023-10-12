import powerxrd as xrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Element
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
from itertools import combinations
import pandas as pd
import ast


def matproject_xrd_to_dict(directory_path):
    """Reads and processes CSV files from the given directory and returns a dictionary of DataFrames."""
    
    # Initialize an empty dictionary to store the DataFrames
    dataframes_dict = {}

    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(directory_path, file)
            
            # Read the file into a DataFrame
            df_temp = pd.read_csv(file_path)
            df_temp['hkl'] = df_temp['hkl'].apply(ast.literal_eval)
            
            # Store the DataFrame in the dictionary with the key as the filename without extension
            key_name = os.path.splitext(file)[0]
            dataframes_dict[key_name] = df_temp

    return dataframes_dict


def download_matproj_structures_and_xrd(api_key, alloy_elements, impurities):
    # Initialize MPRester
    m = MPRester(api_key)

    # Create directories to save the CIF files and XRD spectra
    if not os.path.exists("Filtered_structures"):
        os.makedirs("Filtered_structures")
    if not os.path.exists("XRD_spectra"):
        os.makedirs("XRD_spectra")

    # All elements in the periodic table minus the combined list of alloy_elements and impurities
    excluded_elements = [e.symbol for e in Element if e.symbol not in alloy_elements + impurities]

    total_downloads = 0
    for combo in combinations(alloy_elements, 2):  # Combinations of two alloy elements
        criteria = {
            "elements": {"$all": list(combo), "$nin": excluded_elements},
            "nelements": {"$gte": 2}
        }
        properties = ["material_id", "pretty_formula"]
        results = m.query(criteria, properties)

        # Download and save the CIF structures and XRD spectra
        for result in results:
            material_id = result["material_id"]
            formula = result["pretty_formula"]

            print(f"Processing material: {formula} (ID: {material_id})...")

            # Save CIF structure
            print("  - Downloading CIF structure...")
            structure = m.get_structure_by_material_id(material_id, final=True)
            cif_file_name = os.path.join("Filtered_structures", f"{formula}_{material_id}.cif")
            structure.to(filename=cif_file_name)
            print(f"  - CIF structure saved to {cif_file_name}.")

            # Calculate and save XRD spectrum
            print("  - Calculating XRD spectrum...")
            sga = SpacegroupAnalyzer(structure)
            conventional_structure = sga.get_conventional_standard_structure()
            calculator = XRDCalculator(wavelength="CuKa")
            pattern = calculator.get_pattern(conventional_structure)

            # Convert the pattern to a DataFrame and save as CSV
            xrd_data = {
                "2Theta": pattern.x,
                "Intensity": pattern.y,
                "hkl": pattern.hkls
            }
            df = pd.DataFrame(xrd_data)
            csv_file_name = os.path.join("XRD_spectra", f"{formula}_{material_id}.csv")
            df.to_csv(csv_file_name, index=False)
            print(f"  - XRD spectrum saved to {csv_file_name}.")

            total_downloads += 1

    print(f"\nCompleted! Downloaded {total_downloads} CIF files and XRD spectra.")




def estimate_d_spacing(theta, wavelength, n=1):
    """
    Estimate the interplanar spacing (d) using Bragg's law.
    
    Parameters:
    - theta (float): The Bragg angle in degrees (half of 2θ value).
    - wavelength (float): Wavelength of the incident X-rays.
    - n (int, optional): Order of reflection. Default is 1.
    
    Returns:
    - float: Estimated d-spacing.
    """
    
    # Convert theta to radians
    theta_rad = np.radians(theta)
    
    # Calculate d-spacing using Bragg's law
    d_spacing = (n * wavelength) / (2 * np.sin(theta_rad))
    
    return d_spacing


def funcgauss(x, y0, a, mean, sigma):
    """Gaussian equation."""
    return y0 + (a / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * sigma**2))


def gaussian_fit_parameters_and_cov(x, y):
    """Fit a Gaussian curve to the data and return the parameters and covariance matrix."""
    mean_guess = x[np.argmax(y)]
    sigma_guess = mean_guess - min(x)
    y0_guess = min(y)
    amplitude_guess = max(y) - y0_guess
    
    popt, pcov = curve_fit(funcgauss, x, y, p0=[y0_guess, amplitude_guess, mean_guess, sigma_guess])
    return popt, pcov

def calculate_r2_and_rmse(y_observed, y_predicted):
    """Calculate R^2 and RMSE."""
    residuals = y_observed - y_predicted
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((y_observed - np.mean(y_observed))**2)
    R2 = 1 - (SS_res / SS_tot)
    RMSE = np.sqrt(np.mean(residuals**2))
    return R2, RMSE

def calculate_fwhm_from_sigma(sigma):
    """Calculate FWHM from the Gaussian sigma."""
    return sigma * 2 * np.sqrt(2 * np.log(2))

def scherrer_analysis(dataframe, wavelength_range, K=0.9, lambdaKa=0.15406, visualize=False):
    """Analyze the given dataframe and return Gaussian fit parameters, Scherrer width, R^2, and RMSE."""
    x = dataframe.iloc[:, 0].to_numpy()
    y = dataframe.iloc[:, 1].to_numpy()
    
    # Extract the segment of the data within the given wavelength range
    mask = (x >= wavelength_range[0]) & (x <= wavelength_range[1])
    x_segment = x[mask]
    y_segment = y[mask]
    
    # Get Gaussian fit parameters and covariance matrix
    params, covariance_matrix = gaussian_fit_parameters_and_cov(x_segment, y_segment)
    
    # Predict y values using the Gaussian parameters
    y_predicted = funcgauss(x_segment, *params)
    
    # Calculate FWHM in radians
    FWHM_degrees = calculate_fwhm_from_sigma(params[3])
    FWHM_radians = np.radians(calculate_fwhm_from_sigma(params[3]))
    
    # Calculate Scherrer width
    theta_rad = np.radians(params[2] / 2)
    scherrer_val = K * lambdaKa / (FWHM_radians * np.cos(theta_rad))
    
    # Calculate R^2 and RMSE
    R2, RMSE = calculate_r2_and_rmse(y_segment, y_predicted)
    
    # Visualization
    if visualize:
        plt.plot(x_segment, y_segment, 'm', label='Data')
        plt.plot(x_segment, y_predicted, 'c--', label='Gaussian Fit')
        plt.xlabel('2 $\\theta$')
        plt.legend()
        plt.show()
    
    return {
        "y_shift": params[0],
        "amplitude": params[1],
        "mean": params[2],
        "sigma": params[3],
        "covariance_matrix": covariance_matrix,
        "scherrer_width": scherrer_val,
        "R2": R2,
        "RMSE": RMSE,
        "FWHM_degrees": FWHM_degrees
    }


def calculate_fwhm(x_values, y_values, peak_index, peak_prominence):


    """
    Calculate the Full Width at Half Maximum (FWHM) for a given peak.

    Parameters:
    - x_values (array-like): The x values (2θ values in this context).
    - y_values (array-like): The y values (intensities in this context).
    - peak_index (int): The index of the peak.
    - peak_prominence (float): The prominence of the peak.

    Returns:
    - float: The FWHM value for the peak.
    """

    half_max = y_values[peak_index] - peak_prominence / 2

    # Find the indices where the intensity is equal to the half maximum on both sides of the peak
    left_indices = np.where(y_values[:peak_index] < half_max)[0]
    right_indices = np.where(y_values[peak_index:] < half_max)[0] + peak_index

    if left_indices.size > 0 and right_indices.size > 0:
        left_index = left_indices[-1]
        right_index = right_indices[0]
        fwhm = x_values[right_index] - x_values[left_index]
        return fwhm
    else:
        return np.nan

def extract_peak_properties_advanced(df, truncate=(0, 65), visualize=True):
    """
    Extract peak properties from an XRD spectrum dataframe using advanced peak detection.
    
    Parameters:
    - df (DataFrame): Input dataframe with first column as 2theta and second column as the XRD spectrum.
    - truncate (tuple): A tuple defining the 2θ interval for truncation. Default is (0, 65).
    - visualize (bool): Whether to visualize the peaks on the curve. Default is True.
    
    Returns:
    - DataFrame: A dataframe with columns '2θ (degree)', 'Peak Intensity', 'Prominence', and 'FWHM'.
    """
    
    # Truncate the data based on the provided interval
    truncated_data = df[(df.iloc[:, 0] >= truncate[0]) & (df.iloc[:, 0] <= truncate[1])]
    
    # Applying Savitzky-Golay smoothing filter
    smoothed_intensity = savgol_filter(truncated_data.iloc[:, 1], window_length=11, polyorder=3)
    
    # Peak finding with adjusted parameters
    adjusted_peaks, adjusted_properties = find_peaks(smoothed_intensity, distance=2, prominence=3)
    
    # Extracting peak properties
    peak_properties = pd.DataFrame({
        '2θ (degree)': truncated_data.iloc[adjusted_peaks, 0].values,
        'Peak Intensity': truncated_data.iloc[adjusted_peaks, 1].values,
        'Prominence': adjusted_properties['prominences']
    })

    # Calculate FWHM for each of the adjusted peaks
    fwhms = []
    for peak_index in adjusted_peaks:
        prominence = adjusted_properties['prominences'][np.where(adjusted_peaks == peak_index)[0][0]]
        fwhm = calculate_fwhm(truncated_data.iloc[:, 0].values, smoothed_intensity, peak_index, prominence)
        fwhms.append(fwhm)

    peak_properties['FWHM'] = fwhms
    
    if visualize:
        # Sort the peaks by prominence to get the top 20
        top_20_peaks = peak_properties.sort_values(by='Prominence', ascending=False).head(20)
        
        # Plot the curve and annotate the top 20 peaks
        plt.figure(figsize=(14, 7))
        plt.plot(truncated_data.iloc[:, 0], truncated_data.iloc[:, 1], label="XRD Intensity")
        
        for index, row in top_20_peaks.iterrows():
            plt.plot(row['2θ (degree)'], row['Peak Intensity'], "ro")
            plt.annotate(f"2θ: {row['2θ (degree)']:.2f}\nProminence: {row['Prominence']:.2f}",
                         (row['2θ (degree)'], row['Peak Intensity']), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, verticalalignment='bottom')
        
        plt.title("Top 20 Peaks Visualization on XRD Spectrum")
        plt.xlabel("2θ")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    peak_properties['Peak Intensity Normalized'] = (peak_properties['Peak Intensity'] / peak_properties['Peak Intensity'].max()) * 100
    peak_properties['Prominence Normalized'] = (peak_properties['Prominence'] / peak_properties['Prominence'].max()) * 100

    
    return peak_properties

# Takes in a dataframe with x = 2theta, y = intensity and returns a df with x = 2theta, y1 = intensity, y2 = backsubt_intensity
def backsub_from_dataframe(input_df):
    # Extract data from the input DataFrame
    x = input_df.iloc[:, 0].values
    y = input_df.iloc[:, 1].values
    
    # Create an instance of xrd.Chart with the data
    chart = xrd.Chart(x, y)
    
    # Get the background subtracted data
    _, backsub_y = chart.backsub()
    
    # Combine the data and background subtracted data into a new DataFrame
    df = pd.DataFrame({
        'X': x,
        'Y': y,
        'Backsub_Y': backsub_y
    })
    
    return df

if __name__ == "__main__":

    folder_path = r"input_data\230704-19"

# Check if 'input_data' directory exists in the current working directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} not found")

    # Get a list of all .xy files in the specified folder
    xy_files = glob.glob(f"{folder_path}/*.xy")

    xy_files

# Assuming the backsub_from_dataframe function is already defined as given earlier

    # Iterate over each .xy file
    for filename in xy_files:
        # Read the xy file into a DataFrame
        df = pd.read_csv(filename, delimiter=r"\s+", header=None, names=["X", "Y"])
        
        # Run backsub_from_dataframe on the df
        processed_df = backsub_from_dataframe(df)
        
        # Save the resultant df as a .csv with _bckSub appended to the original filename
        new_filename = filename.replace(".xy", "_bckSub.csv")
        processed_df.to_csv(new_filename, index=False)

    # Return a message indicating the process is complete
    completed_message = "Background subtraction and saving process completed!" if xy_files else "No .xy files found!"
    print(completed_message)


