import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import measure
import copy
import imageio
from scipy.ndimage.interpolation import zoom
import pydicom as pydicom
import os
import numpy as np
import pydicom
import streamlit as st
import matplotlib.pyplot as plt
from skimage.morphology import disk, closing, binary_dilation
from skimage import measure, morphology


# Load the CSV file
def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    return test_df,train_df

test_df,train_df = load_data()
def load_patient_data(train_df):
    patient_ids = ['ID00007637202177411956430', 'ID00009637202177434476278', 'ID00010637202177584971671', 'ID00011637202177653955184', 'ID00012637202177665765362', 'ID00014637202177757139317', 'ID00015637202177877247924', 'ID00019637202178323708467', 'ID00020637202178344345685', 'ID00023637202179104603099', 'ID00025637202179541264076', 'ID00026637202179561894768', 'ID00027637202179689871102', 'ID00030637202181211009029', 'ID00032637202181710233084', 'ID00035637202182204917484', 'ID00038637202182690843176', 'ID00042637202184406822975', 'ID00047637202184938901501', 'ID00048637202185016727717', 'ID00051637202185848464638', 'ID00052637202186188008618', 'ID00060637202187965290703', 'ID00061637202188184085559', 'ID00062637202188654068490', 'ID00067637202189903532242', 'ID00068637202190879923934', 'ID00072637202198161894406', 'ID00073637202198167792918', 'ID00075637202198610425520', 'ID00076637202199015035026', 'ID00077637202199102000916', 'ID00078637202199415319443', 'ID00082637202201836229724', 'ID00086637202203494931510', 'ID00089637202204675567570', 'ID00090637202204766623410', 'ID00093637202205278167493', 'ID00094637202205333947361', 'ID00099637202206203080121', 'ID00102637202206574119190', 'ID00104637202208063407045', 'ID00105637202208831864134', 'ID00108637202209619669361', 'ID00109637202210454292264', 'ID00110637202210673668310', 'ID00111637202210956877205', 'ID00115637202211874187958', 'ID00117637202212360228007', 'ID00119637202215426335765', 'ID00122637202216437668965', 'ID00123637202217151272140', 'ID00124637202217596410344', 'ID00125637202218590429387', 'ID00126637202218610655908', 'ID00127637202219096738943', 'ID00128637202219474716089', 'ID00129637202219868188000', 'ID00130637202220059448013', 'ID00131637202220424084844', 'ID00132637202222178761324', 'ID00133637202223847701934', 'ID00134637202223873059688', 'ID00135637202224630271439', 'ID00136637202224951350618', 'ID00138637202231603868088', 'ID00139637202231703564336', 'ID00140637202231728595149', 'ID00149637202232704462834', 'ID00161637202235731948764', 'ID00165637202237320314458', 'ID00167637202237397919352', 'ID00168637202237852027833', 'ID00169637202238024117706', 'ID00170637202238079193844', 'ID00172637202238316925179', 'ID00173637202238329754031', 'ID00180637202240177410333', 'ID00183637202241995351650', 'ID00184637202242062969203', 'ID00186637202242472088675', 'ID00190637202244450116191', 'ID00192637202245493238298', 'ID00196637202246668775836', 'ID00197637202246865691526', 'ID00199637202248141386743', 'ID00202637202249376026949', 'ID00207637202252526380974', 'ID00210637202257228694086', 'ID00213637202257692916109', 'ID00214637202257820847190', 'ID00216637202257988213445', 'ID00218637202258156844710', 'ID00219637202258203123958', 'ID00221637202258717315571', 'ID00222637202259066229764', 'ID00224637202259281193413', 'ID00225637202259339837603', 'ID00228637202259965313869', 'ID00229637202260254240583', 'ID00232637202260377586117', 'ID00233637202260580149633', 'ID00234637202261078001846', 'ID00235637202261451839085', 'ID00240637202264138860065', 'ID00241637202264294508775', 'ID00242637202264759739921', 'ID00248637202266698862378', 'ID00249637202266730854017', 'ID00251637202267455595113', 'ID00255637202267923028520', 'ID00264637202270643353440', 'ID00267637202270790561585', 'ID00273637202271319294586', 'ID00275637202271440119890', 'ID00276637202271694539978', 'ID00279637202272164826258', 'ID00283637202278714365037', 'ID00285637202278913507108', 'ID00288637202279148973731', 'ID00290637202279304677843', 'ID00291637202279398396106', 'ID00294637202279614924243', 'ID00296637202279895784347', 'ID00298637202280361773446', 'ID00299637202280383305867', 'ID00305637202281772703145', 'ID00307637202282126172865', 'ID00309637202282195513787', 'ID00312637202282607344793', 'ID00317637202283194142136', 'ID00319637202283897208687', 'ID00322637202284842245491', 'ID00323637202285211956970', 'ID00329637202285906759848', 'ID00331637202286306023714', 'ID00335637202286784464927', 'ID00336637202286801879145', 'ID00337637202286839091062', 'ID00339637202287377736231', 'ID00340637202287399835821', 'ID00341637202287410878488', 'ID00342637202287526592911', 'ID00343637202287577133798', 'ID00344637202287684217717', 'ID00351637202289476567312', 'ID00355637202295106567614', 'ID00358637202295388077032', 'ID00360637202295712204040', 'ID00364637202296074419422', 'ID00365637202296085035729', 'ID00367637202296290303449', 'ID00368637202296470751086', 'ID00370637202296737666151', 'ID00371637202296828615743', 'ID00376637202297677828573', 'ID00378637202298597306391', 'ID00381637202299644114027', 'ID00383637202300493233675', 'ID00388637202301028491611', 'ID00392637202302319160044', 'ID00393637202302431697467', 'ID00398637202303897337979', 'ID00400637202305055099402', 'ID00401637202305320178010', 'ID00405637202308359492977', 'ID00407637202308788732304', 'ID00408637202308839708961', 'ID00411637202309374271828', 'ID00414637202310318891556', 'ID00417637202310901214011', 'ID00419637202311204720264', 'ID00421637202311550012437', 'ID00422637202311677017371', 'ID00423637202312137826377', 'ID00426637202313170790466']
    patient_ids = sorted(patient_ids)

    no_of_instances = [30, 394, 106, 31, 49, 31, 295, 29, 493, 27, 24, 239, 358, 433, 205, 574, 346, 497, 103, 26, 122, 311, 275, 253, 30, 319, 118, 24, 355, 320, 64, 94, 1018, 266, 30, 36, 70, 37, 24, 25, 233, 498, 53, 512, 300, 211, 302, 24, 30, 71, 71, 258, 398, 22, 17, 50, 48, 27, 57, 210, 407, 52, 451, 337, 404, 304, 64, 271, 245, 28, 12, 221, 253, 115, 178, 31, 602, 577, 56, 62, 408, 361, 245, 121, 102, 87, 825, 64, 303, 268, 18, 21, 64, 375, 465, 58, 54, 207, 347, 17, 67, 405, 38, 296, 33, 253, 18, 16, 319, 28, 30, 61, 63, 74, 284, 25, 30, 306, 31, 26, 240, 485, 35, 31, 33, 62, 61, 29, 469, 409, 521, 38, 106, 258, 260, 74, 30, 33, 64, 56, 30, 60, 27, 39, 201, 29, 37, 312, 291, 33, 66, 266, 341, 217, 364, 396, 346, 423, 32, 429, 56, 478, 49, 265, 24, 30, 54, 36, 278, 250, 56, 28, 62, 473, 290, 408]
    age = []
    sex = []
    smoking_status = []

    for patient_id in patient_ids:
        patient_info = train_df[train_df['Patient'] == patient_id].reset_index()
        #no_of_instances.append(len(os.listdir(train_dir + patient_id)))
        age.append(patient_info['Age'][0])
        sex.append(patient_info['Sex'][0])
        smoking_status.append(patient_info['SmokingStatus'][0])

    patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex, smoking_status)), 
                             columns=['Patient', 'no_of_instances', 'Age', 'Sex', 'SmokingStatus'])

    return patient_df


# Create a Streamlit app to display the patient information


st.title("Pulmonary Fibrosis Project")
st.write("Hello! I'm Sakshi. Welcome to my Pulmonary Fibrosis Project Demo. Here, I will take you through EDA, data analysis and preprocessing that I did which helped me get the desired results through my ML Algorithm. ")

st.subheader("Total Patients")
st.write("First of all, let's see the amount of data we have. The total number of patients we have data on are: ")
st.write("Total Patients in Train set: ",train_df['Patient'].count())
st.write("Total Patients in Test set: ",test_df['Patient'].count())
# Display the dataset

st.subheader("Filtering Data")
st.write("Now, let's look at our tabular dataset. Feel free to play around with the sidebar. The dataframe below will be manipulated according to your selection.")
# Sidebar filters
st.sidebar.title("Filtered Training Data")
selected_age = st.sidebar.slider("Select Age Range", min_value=int(train_df["Age"].min()), max_value=int(train_df["Age"].max()),value=70)
selected_sex = st.sidebar.selectbox("Select Sex", train_df["Sex"].unique())
selected_smoking_status = st.sidebar.selectbox("Select Smoking Status", train_df["SmokingStatus"].unique(),)

filtered_data = train_df[
    (train_df["Age"] <= selected_age) &
    (train_df["Sex"] == selected_sex) &
    (train_df["SmokingStatus"] == selected_smoking_status)
]

# Display filtered train_df
st.write("Filtered training data")
st.write(filtered_data)

st.write("According to the selection above, we can also see the distribution of FVC for the dataframe")
# train_df Visualization (you can add more plots and analysis)
st.write("Visualization of the Filtered data")
st.bar_chart(filtered_data["FVC"])

st.subheader("Grouped Patient Data")
st.write("Now, let's group the data based on patients. Here we can see the number of data points we have on a patient, their age, sex and smoking habits.")
patient_data = load_patient_data(train_df)
st.write("Patient Data")
st.dataframe(patient_data)


st.subheader("Feature Distribution Viewer")
st.write("Choose any feature of your choice to view it's distribution")
# Dropdown for selecting a feature
selected_feature = st.selectbox("Select a feature:", train_df.columns,index = 1)

# Create a sub-dataframe for the selected feature
selected_data = train_df[selected_feature]

# Calculate value counts and convert it to a DataFrame
value_counts_df = pd.DataFrame({'Counts': selected_data.value_counts().values}, index=selected_data.value_counts().index)

# Display a bar chart of value counts
st.bar_chart(value_counts_df)

st.subheader("Interactive Data Visualization")
st.write("Let's check how these features are correlated to one another. Feel free to choose the features for the X and Y Axes")
# Dropdown for selecting the x and y axes
x_axis = st.selectbox("Select X-axis:", train_df.columns,index=2)
y_axis = st.selectbox("Select Y-axis:", train_df.columns,index=1)

# Create a scatter plot based on user selections
scatter_plot = alt.Chart(train_df).mark_circle(size=60).encode(
    x=x_axis,
    y=y_axis,
    tooltip=[x_axis, y_axis]
).interactive()
st.altair_chart(scatter_plot, use_container_width=True)


st.subheader("Pulmonary Fibrosis Patient Information")
st.write("Now, let's dive into the image database. Let's have a look at the DICOM Images. Feel free to choose any patient ID from the dropdown to see tabular and image data along with some image statistics")
train_dir = './data'
folder_names = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]

# Show the folder names in the dropdown

selected_patient = st.selectbox("Select a patient:", folder_names)
#selected_patient = st.selectbox("Select a patient:", train_df['Patient'].unique())
patient_info = train_df[train_df['Patient'] == selected_patient]

st.write(patient_info)

# Task 2: Display pydicom images for the selected patient

def show_first_n_images(patient_id, train_dir, n=9):
# Get a list of pydicom files for the selected patient
    patient_folder = os.path.join(train_dir, patient_id)
    dicom_files = [f for f in os.listdir(patient_folder) if f.endswith('.dcm')]

    # Randomly select the first 9 images or fewer if there are not enough available
    num_images = min(n, len(dicom_files))
    selected_files = random.sample(dicom_files, num_images)

    # Create a 3x3 grid to display the selected images
    rows = 3
    cols = 3
    fig = plt.figure(figsize=(12, 12))

    for i, image_filename in enumerate(selected_files):
        if i >= n:
            break

        dicom_path = os.path.join(patient_folder, image_filename)
        dicom_image = pydicom.dcmread(dicom_path)

        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(dicom_image.pixel_array, cmap='gray')
        plt.title(f'Image {i + 1}')
        plt.axis('off')

    fig.suptitle(f'CT Scan Images for Patient: {patient_id}', fontsize=16)
    st.pyplot(fig)
show_first_n_images(selected_patient,train_dir)

# Show pydicom image statistics
def load_scan(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def set_lungwin(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg

patient_folder = os.path.join(train_dir, selected_patient)
scan2 = load_scan(patient_folder)
scan_array2 = set_lungwin(get_pixels_hu(scan2))
if len(scan2) > 0:
    st.write("**DICOM Image Statistics:**")
    st.write(f"Number of pydicom images: {len(scan2)}")
    st.write(f"Image dimensions: {scan_array2[0].shape}")
    st.write(f"Pixel spacing: {scan2[0].PixelSpacing}")
    st.write(f"Slice thickness: {scan2[0].SliceThickness}")

# Function to load pydicom images and segment lungs


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None



# Folder containing pydicom images
dicom_folder = './data'
actual_path=os.path.join(dicom_folder, selected_patient)
scans = load_scan(actual_path)
scan_array = set_lungwin(get_pixels_hu(scans))
# Load pydicom images and process lung segmentation
# if st.button("Load and Process pydicom Images"):
    
#     st.success("pydicom images loaded and processed.")

#     # Display the first image from the processed scans
#     if len(scans) > 0:
#         st.image(scan_array[0], use_column_width=True, channels='GRAY', caption="Processed pydicom Image")

# Save processed images as a GIF
 



# Sidebar for lung segmentation parameters
patient_ids = os.listdir('./data')
patient_id = 'ID00267637202270790561585'
dicom_filenames = os.listdir('./data/' + patient_id)
dicom_paths = ['./data/' + patient_id + '/'+ file for file in dicom_filenames]
    
def load_scan(paths):
    slices = [pydicom.read_file(path ) for path in paths]
    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness 
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)   
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


patient_dicom = load_scan(dicom_paths)
patient_pixels = get_pixels_hu(patient_dicom)
#sanity check
st.subheader("DICOM Image Analysis and Statistics")

# Display the sanity check plot using st.pyplot
st.write("Let's look at Patient ID00267637202270790561585. Let's look at a slice from their CT Scan images")
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(patient_pixels[46], cmap=plt.cm.bone)
ax.axis(False)
st.pyplot(fig)


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8)+1
    labels = measure.label(binary_image)
 
    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the  patient
    # More resistant to “trays” on which the patient lays cutting the air around the person in half
    background_label = labels[0,0,0]
 
    # Fill the air around the person
    binary_image[background_label == labels] = 2
 
    # Method of filling the lung structures (that is superior to 
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
 
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
 
    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

# get masks 
segmented_lungs = segment_lung_mask(patient_pixels, fill_lung_structures=False)
segmented_lungs_fill = segment_lung_mask(patient_pixels, fill_lung_structures=True)
internal_structures = segmented_lungs_fill - segmented_lungs
copied_pixels = copy.deepcopy(patient_pixels)
for i, mask in enumerate(segmented_lungs_fill): 
    get_high_vals = mask == 0
    copied_pixels[i][get_high_vals] = 0
seg_lung_pixels = copied_pixels



# Display the original and segmented images side by side
st.write("Now, let's try to segment the lungs from the original image. Here's the Original and Segmented Images:")
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(patient_pixels[46], cmap=plt.cm.bone)
ax[0].axis(False)
ax[0].set_title('Original')
ax[1].imshow(seg_lung_pixels[46], cmap=plt.cm.bone)
ax[1].axis(False)
ax[1].set_title('Segmented')

# Use st.pyplot() to display Matplotlib plot in Streamlit
st.pyplot(fig)

slice_id = 46

# Display images using st.pyplot
st.write("Now, let's try to apply a lung mask on the original image. We can then segment this image")
st.write("Original Dicom, Lung Mask, Segmented Lung, Segmentation with Internal Structure:")
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(patient_pixels[slice_id], cmap=plt.cm.bone)
ax[0, 0].set_title('Original Dicom')
ax[0, 0].axis(False)

ax[0, 1].imshow(segmented_lungs_fill[slice_id], cmap=plt.cm.bone)
ax[0, 1].set_title('Lung Mask')
ax[0, 1].axis(False)

ax[1, 0].imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)
ax[1, 0].set_title('Segmented Lung')
ax[1, 0].axis(False)

ax[1, 1].imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)
ax[1, 1].imshow(internal_structures[slice_id], cmap='jet', alpha=0.7)
ax[1, 1].set_title('Segmentation with Internal Structure')
ax[1, 1].axis(False)
st.pyplot(fig)

st.subheader("Everyone loves GIFs")
st.write("Let's try to look at the images but with animated GIFs to understand them better.")
imageio.mimsave("/tmp/gif.gif", scan_array, loop=4, duration = 0.3)
st.image("/tmp/gif.gif", use_column_width=True)

st.subheader("Lung Segmentation Mask")

st.image('./1.png',use_column_width=True)

st.subheader("Segmented Part of Lung Tissue")

st.image('./2.png',use_column_width=True)


original_image = patient_pixels[46]
original = segmented_lungs_fill[46]
#rows, cols = 3, 3  # Define the number of rows and columns

# Streamlit app
st.subheader("Morphological Operations")
st.write("Now, let's look at the segmented part of the lung tissue frame by frame to understand segmentation on deeper level.")
output_dir = 'processed_images'
os.makedirs(output_dir, exist_ok=True)

# Check if images are already processed
processed_images = []
rows = 4
cols = 4
for i in range(rows * cols):
    image_path = os.path.join(output_dir, f'slice_{i*2+25}.png')
    if os.path.exists(image_path):
        processed_images.append(plt.imread(image_path))
    else:
        image = patient_pixels[i*2+25] * binary_dilation(closing(segmented_lungs_fill[i*2+25], disk(20)), disk(4))
        plt.imsave(image_path, image, cmap='gray')
        processed_images.append(image)

# Create a Streamlit figure
fig, ax = plt.subplots(rows, cols, figsize=(15, 15))

# Populate the subplots
for i in range(rows * cols):
    ax[int(i/rows), int(i % rows)].set_title(f'slice({i*2+25})')
    ax[int(i/rows), int(i % rows)].imshow(processed_images[i], cmap=plt.cm.gray)
    ax[int(i/rows), int(i % rows)].axis('off')

# Display the figure in Streamlit
st.pyplot(fig)
