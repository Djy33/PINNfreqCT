Data Description 

Data Folder Structure:

Data is divided into folders based on subject number and day of the experiment. Data is collected for each subject on a single day except subject 1 which has data for 2 different days (day1 and day4) to test the repeatability of blood pressure prediction after several days as shown in Table S1. Each subject data is divided into sub-folders based on setups (baseline, HGCP, ...etc) and index to show the order of the setups. There are multiple folders for the same setup with a different index which are repeated trials for the same setup. Each setup folder consists of the data files in CSV format called data_trial. Each data type is in a separate data_trial CSV file. Each data_trial CSV file for a specific data type is segmented into multiple files with indexes 01,02,..etc to avoid a large file size. The data in data_trial CSV files were collected successively after each other based on the sequence of the index. Data within the data_trial CSV file is arranged in columns. The first column is timestamps in seconds and the rest of the columns are the data.

Data Types:

*bioz data = BioZ1 (radial artery closer to wrist), BioZ2 (radial artery closer to the heart), BioZ3 (ulnar artery closer to wrist), BioZ4 (ulnar artery closer to the heart). Sampling Rate = 1250 Hz
*finapresBP = Continuous BP measurement with Finapres representing brachial BP. Sampling Rate = 200 Hz
*finapresPPG = PPG measurement from Finapres on fingertip. Sampling Rate = 75Hz
*ppg = PPG measurement from fingertip with BioZ XL board. Sampling Rate = 1250 Hz
BioZ signals are in mOhms, PPG signals are a.u., BP signal is in mmHg, signals are synchronized in time. 

Setups Definition: 

(a)	HGCP (Hand Grip Cold Pressor): the subject performs a handgrip (HG) exercise for 3 minutes, slowly raising their DBP and SBP, then placing their hand into an ice cold water bucket (cold pressor, CP) for 1 minute to ensure that BP first goes even higher, then very slowly decreases over the 4 minute resting period.
(b)	Cycling: the subject is stationed to perform a set of bike cycling treadmill exercises for 4 minutes, with 4 minutes of break for resting in between. 
(c)	Valsalva: session with multiple Valsalva maneuvers. Each Valsalva maneuver consists of a subject pinching their nose while trying to breathe out intensely for 20-30 seconds, creating an extensive buildup of inner pressure, both raising BR, then decreasing, and rapidly increasing it once again very rapidly. 
(d)	baseline: At the beginning of data collection with no BP change protocol, participants are at rest.
(e)	rest: no BP change protocol, participants are at rest. 
