import matplotlib.pyplot as plt
import pydicom 
import re

def plot_single_dicom_image(image_path, study_id = None, figsize = (20,10)):
    """
	image_path: path to .dcm image
	figsize:    customizable figure size
	"""
    try:
        x = pydicom.dcmread(image_path)
    except OSError as e:
        print(
            "Can't read jpg: %s\n"
            "Received error: %s"
            "" % (image_path, e)
        )
        return

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    ax.imshow(x.pixel_array,  cmap = plt.cm.bone)
    if study_id is not None:
    	ax.set_title(f'Study ID: {study_id} \nPatient ID: {x.PatientID}\n StudyDate: {x.StudyDate}')
    else:
    	ax.set_title(f'Patient ID: {x.PatientID}\n StudyDate: {x.StudyDate}')

def display_study(study_id, overall_dataframe):
    """
    study_id: a unique identifier for a study done on a patient
              each study_id may have more than one image
    overall_dataframe: the dataframe containing the DICOM file paths
    """
    # Get the record
    record = overall_dataframe.loc[overall_dataframe.study_id == study_id]
    image_paths = record.path
    for path in record.path:
        plot_single_dicom_image(path, study_id)
    return None

def get_study_image_pixels(study_id, overall_dataframe):
    """
    study_id: a unique identifier for a study done on a patient
              each study_id may have more than one image
    overall_dataframe: the dataframe containing the DICOM file paths
    """
    record = overall_dataframe.loc[overall_dataframe.study_id == study_id]
    image_paths = record.path
    study_dict ={}
    for (image_number, path) in enumerate(record.path):
        try:
            x = pydicom.dcmread(path)
        except OSError as e:
            print(
                "Can't read jpg: %s\n"
                "Received error: %s"
                "" % (path, e)
            )
            return
        study_dict[image_number] = x.pixel_array

    return study_dict

def get_report_text(study_id, overall_dataframe):
    """
    study_id: a unique identifier for a study done on a patient
              each study_id will have one report associated with it (report might relate to multiple images)
    overall_dataframe: the dataframe containing the report file paths
    """
    # Get the report
    record = overall_dataframe.loc[overall_dataframe.study_id == study_id]
    image_paths = record.path.iloc[0]
    study_id = str(study_id)

    report_path = re.sub(str('s' + study_id + '/.*$'), str('s' + study_id + '.txt'), str(image_paths))

    report_text = ''
    with open(report_path, 'r') as file:
        report_text = file.read().replace('\n', '')

    return report_text