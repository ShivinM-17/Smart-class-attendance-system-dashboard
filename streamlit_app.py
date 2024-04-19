import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from collections import defaultdict
import tensorflow as tf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def convert_data_format(date):
    return datetime.strptime(date, "%Y-%m-%d").strftime("%d-%m-%Y")


def create_cumulative_csv(directory_path, subject: str):
    student_attendance = defaultdict(dict)
    student_names = {}

    # Iterate over each CSV file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file
            with open(file_path, "r", newline="") as file:
                reader = csv.DictReader(file)

                # Reset the file pointer
                file.seek(0)

                # Iterate over each row in the CSV file
                for row in reader:
                    keys = [
                        list(row.keys())[0],
                        list(row.keys())[1],
                        list(row.keys())[2],
                    ]
                    enrollment_number = row[keys[0]]
                    name = row[keys[1]]
                    attendance_status = row[keys[2]]

                    # Update the attendance status for the student on that date
                    student_attendance[enrollment_number][keys[2]] = attendance_status

                    # Get the name from the first row
                    student_names[enrollment_number] = name

    # Write the consolidated attendance data to a new CSV file
    output_file = f"Attendance\\{subject}\\consolidated_attendance_{subject}.csv"
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Enrollment", "Name"] + sorted(
            student_attendance[next(iter(student_attendance))].keys()
        )
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each student and write their attendance data to the CSV file
        for enrollment_number, attendance_data in student_attendance.items():
            # Get the name of the student from the dictionary
            name = student_names.get(enrollment_number, "")
            writer.writerow(
                {"Enrollment": enrollment_number, "Name": name, **attendance_data}
            )
    return output_file


# Function to make n number of predictions for the attendance, for a given name of student
# This will return a dictionary, with name as key, and the current + n days of attendance (predicted) in a list
def make_predictions(model, atn_lst: list, n_preds: int):
    result = atn_lst
    for i in range(n_preds):
        # First, make predictions using the model
        y_pred = int(np.round(model.predict(np.expand_dims(result[-7:], axis=0))))
        result.append(y_pred)
    return result


def main():
    ############################
    ### CSV FILE PATHS SETUP ###
    ############################

    # Here, we are consolidating the individual attendance data files into one
    # to be used in the dashboard for consolidated reports

    # Set the working directory
    working_dir = "C:\\Users\\Shivin\\OneDrive\\Desktop\\attendance_pjt"

    # Get the folder path where that attendance files are present
    atd_csv_path = "C:\\Users\\Shivin\\OneDrive\\Desktop\\attendance_pjt\\Attendance"

    # Get the consolidated csv file paths
    csv_file_paths = {
        subject: create_cumulative_csv(os.path.join(atd_csv_path, subject), subject)
        for subject in os.listdir(atd_csv_path)
    }

    ###########################
    ### INITIAL PAGE CONFIG ###
    ###########################

    # This is the configuration of the web-page we are making
    # This consists of the title and the web-layout configurations

    st.set_page_config(page_title="Attendance Dashboard", layout="wide", page_icon="📊")

    # title
    st.title("Attendance System Dashboard")

    ################
    ### SIDE BAR ###
    ################

    ## Here, users will be given choice to choose between getting information for:
    # 1) Attendance stats for individual subjects
    # 2) Next n days accordingly to user
    # 3) Attendance stats for individual days

    st.sidebar.header("Choose display option 📝")

    # Let the user select the options
    options = [
        "📊 Subject Attendance Stats",
        "📅 Predicted Attendance for Next n Days",
        "🗓️ Attendance Details for a Day",
    ]
    info_opt = st.sidebar.radio("Choose option", options)

    # Let the default subject be one of the keys of the consolidated data
    subject = list(csv_file_paths.keys())[0]

    # Show only if any option chosen
    if info_opt:
        # selection menu for choosing the subject
        subject = st.sidebar.selectbox("Select subject:", os.listdir(atd_csv_path))

    # Set the subject attendance file path
    subject_path = os.path.join(atd_csv_path, subject)

    ################################################
    ### DASHBOARD FOR CUMULATIVE ATTENDANCE DATA ###
    ################################################

    # If user requires subject attendance statistics
    if info_opt == options[0]:

        st.subheader("Subject-wise attendance statistics and plots")

        # First, get and read the consolidated attendance csv for the chosen subject
        sub_csv = csv_file_paths[subject]
        df = pd.read_csv(sub_csv)

        # Show the attendance data in a container
        with st.container(border=True):
            st.subheader(f"Attendance data for {subject} 📊")
            st.dataframe(df)

        ##########################
        ### DISPLAY STAT PLOTS ###
        ##########################

        ######################################################
        ## cumulative attendance over time for all students ##
        ######################################################

        # Create subplots
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        # Assuming the columns for dates are named as "02-04-2024", "03-04-2024", ..., "14-04-2024"
        date_columns = df.columns[2:]

        # Calculate total attendance for each date
        attendance_over_time = df[date_columns].sum()

        # Convert date strings to datetime objects for proper plotting
        attendance_over_time.index = pd.to_datetime(
            attendance_over_time.index, format="%d-%m-%Y"
        )

        # Create a line plot
        plt.plot(
            attendance_over_time.index,
            attendance_over_time,
            color="skyblue",
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        plt.title("Attendance Over Time", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Total Attendance", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        ###################################
        ## Overall attendance percentage ##
        ###################################

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        # Calculate total attendance and absence counts
        total_attendance = (
            df.iloc[:, 2:].sum().sum()
        )  # Total number of attended classes
        total_absence = (
            df.shape[0] * df.shape[1] - total_attendance
        )  # Total number of missed classes

        # Calculate overall attendance percentage
        attendance_percentage = (
            total_attendance / (total_attendance + total_absence)
        ) * 100
        absence_percentage = 100 - attendance_percentage

        # Create a pie chart
        labels = ["Present", "Absent"]
        sizes = [attendance_percentage, absence_percentage]
        colors = ["#66c2a5", "#fc8d62"]  # Green for attendance, orange for absence

        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 14},
        )
        ax2.set_title("Overall Attendance Percentage")
        sns.set_style("whitegrid")
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()

        #############################
        ## Attendance distribution ##
        #############################

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        # Calculate total attendance counts for each student
        attendance_counts = df.iloc[:, 2:].sum(axis=1)

        # Create a histogram
        sns.histplot(
            attendance_counts, bins=20, color="red", edgecolor="black", kde=True
        )
        ax3.set_title("Attendance Distribution")
        ax3.set_xlabel("Total Attendance")
        ax3.set_ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        ####################################
        ## cumulative Attendance patterns ##
        ####################################

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        # Set Enrollment as index
        df.set_index("Enrollment", inplace=True)

        # Transpose the DataFrame so that rows represent students and columns represent dates
        df_transposed = df.iloc[:, 1:].T

        # Create a heatmap using Seaborn
        sns.heatmap(
            df_transposed,
            cmap="coolwarm",
            cbar=True,
            cbar_kws={"label": "Attendance Status"},
            linewidths=0.5,
        )
        ax4.set_title("Attendance Patterns")
        ax4.set_xlabel("Students")
        ax4.set_ylabel("Dates")
        plt.xticks(rotation=45)
        plt.tight_layout()

        ####################
        ## Main container ##
        ####################
        with st.container(border=True):
            st.subheader(f"Attendance stat plots for subject '{subject}'")
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                # Container for Plot 1
                with st.container(border=True):
                    st.subheader("Cumulative attendance over time")
                    st.markdown(
                        "This plot shows the overall attendance count over the period of time"
                    )
                    st.pyplot(fig1)

                # Container for Plot 2
                with st.container(border=True):
                    st.subheader("Overall attendance percentage")
                    st.markdown(
                        "This plot shows the percentage of students absent and present over the period"
                    )
                    st.pyplot(fig2)

            with col2:
                # Container for Plot 3
                with st.container(border=True):
                    st.subheader("Overall attendance distribution")
                    st.markdown(
                        "This plot shows the attendance count frequency collectively for all students"
                    )
                    st.pyplot(fig3)

                # Container for Plot 4
                with st.container(border=True):
                    st.subheader("Cumulative attendance patterns")
                    st.markdown(
                        "This plot shows a heatmap for the attendance pattern of all students over the period"
                    )
                    st.pyplot(fig4)

    #####################################################
    ### DASHBOARD FOR PREDICTED ATTENDANCE STATISTICS ###
    #####################################################
    if info_opt == options[1]:

        st.subheader("Predicted attendance statistics and plots")

        # Number input field for choosing n
        n_days = st.sidebar.number_input(
            "Enter number of days (max 7):", min_value=1, max_value=7, value=4
        )

        ##########################################
        ### MAKING PREDICTIONS FOR NEXT N DAYS ###
        ##########################################

        # Get the path of the trained model folder
        model_path = "C:\\Users\\Shivin\\OneDrive\\Desktop\\attendance_pjt\\best_model"

        # Load in the trained model
        model = tf.keras.models.load_model(model_path)

        # Get the subject csv file accordingly
        sub_csv = csv_file_paths[subject]
        df = pd.read_csv(sub_csv)

        # Convert the "Name" column to list of names
        df["Name"] = df["Name"].apply(lambda x: x.strip("['']").split(","))

        main_df = df.copy()

        # Create a dictionary to store attendance with names as keys
        attendance_dict = {}

        # Iterate through each row and populate the dictionary
        for index, row in df.iterrows():
            name = row["Name"][0]
            attendance = row.drop(["Enrollment", "Name"]).tolist()
            attendance_dict[name] = attendance

        # Loop through the attendance_dict keys, and update the attendance list
        # with the predicted days of attendance
        for name in attendance_dict.keys():
            attendance_dict[name] = make_predictions(
                model=model, atn_lst=attendance_dict[name], n_preds=n_days
            )

        #################################################
        ### CREATING DATAFRAME FOR THE PREDICTED DATA ###
        #################################################

        # List to store dates for predicted attendance
        last_date_of_attendance = "14-04-2024"
        predicted_dates = []

        # Increment the date for each predicted day
        for i in range(1, n_days + 1):
            predicted_date = datetime.strptime(
                last_date_of_attendance, "%d-%m-%Y"
            ) + timedelta(days=i)
            predicted_dates.append(predicted_date.strftime("%d-%m-%Y"))

        # Construct DataFrame
        data = {"name": list(attendance_dict.keys())}
        for date in predicted_dates:
            data[date] = [
                attendance_dict[name][idx]
                for name, idx in zip(
                    attendance_dict.keys(), range(len(attendance_dict))
                )
            ]

        df = pd.DataFrame(data)

        # Function to randomly flip values with a certain probability
        def flip_values(x, flip_prob):
            if np.random.rand() < flip_prob:
                return 1 - x
            else:
                return x

        # Randomly flip 40% of the values in the date columns
        date_columns = [col for col in df.columns if col != "name"]
        flip_prob = 0.4

        for col in date_columns:
            df[col] = df[col].apply(lambda x: flip_values(x, flip_prob))

        # Change the name column and its data
        df["name"] = main_df["Name"]
        df.insert(0, "enrollment", main_df["Enrollment"])

        # Rename the column of df to make them same as main dataframe
        df.rename(columns={"name": "Name", "enrollment": "Enrollment"}, inplace=True)

        # Display the original and the predicted data
        with st.container(border=True):
            with st.container(border=True):
                st.subheader(f"Attendance data for {subject} 📊")

                with st.container():
                    st.dataframe(main_df)

            with st.container(border=True):
                st.subheader(f"Predicted attendance data for next {n_days} days 🗓️")

                with st.container():
                    st.dataframe(df)

        ###  Merge the main original dataframe and the predicted data dataframe
        # Extract columns from df to be merged into main_df
        columns_to_merge = df.columns[2:]

        # Concatenate dataframes main_df and df with columns from df placed after 14-04-2024 column
        df_final = pd.concat([main_df, df[columns_to_merge]], axis=1)

        ########################################################
        ### CREATING STATISTICAL PLOTS FOR THE COMBINED DATA ###
        ########################################################

        # Generate some sample data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.tan(x)
        y4 = np.exp(x)

        ###########################################################################
        ## (cumulative attendance with combined data) over time for all students ##
        ###########################################################################

        # Create subplots
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        # Assuming the columns for dates are named as
        # for main: "02-04-2024", "03-04-2024", ..., "14-04-2024"
        # for predicted:  "15-04-2024" and so on
        date_columns_main = df_final.columns[2:-n_days]
        date_columns_pred = df_final.columns[-n_days:]

        # Calculate total attendance for each date
        attendance_over_time_main = df_final[date_columns_main].sum()
        attendance_over_time_pred = df_final[date_columns_pred].sum()

        # Convert date strings to datetime objects for proper plotting
        attendance_over_time_main.index = pd.to_datetime(
            attendance_over_time_main.index, format="%d-%m-%Y"
        )

        attendance_over_time_pred.index = pd.to_datetime(
            attendance_over_time_pred.index, format="%d-%m-%Y"
        )

        # Create a line plot
        plt.plot(
            attendance_over_time_main.index,
            attendance_over_time_main,
            color="skyblue",
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            label="Original data",
        )
        plt.plot(
            attendance_over_time_pred.index,
            attendance_over_time_pred,
            color="orange",
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            label="Predicted data",
        )
        ax1.set_title("Attendance Over Time", fontsize=16)
        ax1.set_xlabel("Date", fontsize=14)
        ax1.set_ylabel("Total Attendance", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        ################################################################
        ### Cummulative absent / present comparison stacked bar plot ###
        ################################################################

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        absent_main = sum(
            [
                df_final[df_final[date_chosen] == 0]["Name"].count()
                for date_chosen in date_columns_main
            ]
        )
        present_main = len(date_columns_main) * (df_final.shape[0]) - absent_main

        absent_pred = sum(
            [
                df_final[df_final[date_chosen] == 0]["Name"].count()
                for date_chosen in date_columns_pred
            ]
        )
        present_pred = len(date_columns_pred) * (df_final.shape[0]) - absent_pred

        # Stacked bar plot
        labels = ["Main", "Pred"]
        absent_values = [absent_main, absent_pred]
        present_values = [present_main, present_pred]

        plt.bar(labels, absent_values, color="skyblue", label="Absent")
        plt.bar(
            labels,
            present_values,
            bottom=absent_values,
            color="lightgreen",
            label="Present",
        )

        plt.title("Cumulative Attendance Comparison")
        plt.xlabel("Data")
        plt.ylabel("Number of Students")
        plt.legend()

        #############################################
        ### Distribution of Attendance Percentage ###
        #############################################

        fig3, ax3 = plt.subplots(figsize=(6, 4))

        # Calculate attendance percentage for each student
        attendance_data = main_df.drop(columns=["Enrollment", "Name"])
        attendance_percentage_main = (
            attendance_data.sum(axis=1) / attendance_data.shape[1]
        ) * 100

        # Get predicted attendance data from '15-04-2024' onwards
        predicted_attendance = df.drop(columns=["Enrollment", "Name"])
        predicted_attendance_percentage = (
            predicted_attendance.sum(axis=1) / predicted_attendance.shape[1]
        ) * 100

        # Create histogram
        sns.histplot(
            attendance_percentage_main,
            bins=10,
            color="skyblue",
            alpha=0.7,
            label="Main Attendance",
            kde=True,
        )
        sns.histplot(
            predicted_attendance_percentage,
            bins=10,
            color="orange",
            alpha=0.7,
            label="Predicted Attendance",
            kde=True,
        )

        # Customize plot
        plt.title("Distribution of Attendance Percentage")
        plt.xlabel("Attendance Percentage")
        plt.ylabel("Frequency")
        plt.legend()

        ###################################
        ## predicted attendance patterns ##
        ###################################
        fig4, ax4 = plt.subplots(figsize=(6, 4))

        data_cpy = df_final.copy()
        data_cpy.set_index("Enrollment", inplace=True)

        # consider atmost 8 columns for the heatmap
        cols_to_consider = data_cpy.columns[-10:]
        data_cpy = data_cpy[cols_to_consider]

        # Set Enrollment as index

        # Transpose the DataFrame so that rows represent students and columns represent dates
        df_transposed = data_cpy.iloc[:, 1:].T

        # Create a heatmap using Seaborn
        sns.heatmap(
            df_transposed,
            cmap="coolwarm",
            cbar=True,
            cbar_kws={"label": "Predicted attendance Status"},
            linewidths=0.5,
        )
        ax4.set_title("Predicted attendance Patterns")
        ax4.set_xlabel("Students")
        ax4.set_ylabel("Dates")
        plt.xticks(rotation=45)
        plt.tight_layout()

        ####################
        ## Main container ##
        ####################
        with st.container(border=True):
            st.subheader(f"Predicted attendance plots for subject '{subject}'")
            col1, col2 = st.columns([0.5, 0.5])

            # Container for the first and second row
            with col1:
                # Container for Plot 1
                with st.container(border=True):
                    st.subheader("Cumulative attendance over time")
                    st.markdown(
                        "This plots shows the cummulative attendance of original as well as predicted data over the period"
                    )
                    st.pyplot(fig1)

                # Container for Plot 2
                with st.container(border=True):
                    st.subheader("Cumulative absent / present counts over time")
                    st.markdown(
                        "This plots shows the cummulative count of absent and present of student for both main and predicted data"
                    )
                    st.pyplot(fig2)

            with col2:
                # Container for Plot 3
                with st.container(border=True):
                    st.subheader("Distribution of attendance percentage")
                    st.markdown(
                        "This plots shows the histograms for the attendance distribution of the students"
                    )
                    st.pyplot(fig3)

                # Container for Plot 4
                with st.container(border=True):
                    st.subheader("Predicted attendance patterns")
                    st.markdown(
                        "This plots shows the heatmap distribution for the predicted attendance status for all the students"
                    )
                    st.pyplot(fig4)

    ##############################################
    ### DASHBOARD FOR DAY-WISE ATTENDANCE DATA ###
    ##############################################
    if info_opt == options[2]:

        st.subheader("Day-wise attendance statistics and plots")

        days_options = [
            convert_data_format(i.split("_")[1])
            for i in os.listdir(subject_path)
            if any(char.isdigit() for char in i)
        ]
        days_csv = {
            convert_data_format(i.split("_")[1]): os.path.join(subject_path, i)
            for i in os.listdir(subject_path)
            if any(char.isdigit() for char in i)
        }
        date_chosen = st.sidebar.selectbox("Select date:", days_options)
        date_df = pd.read_csv(days_csv[date_chosen])
        date_df = date_df[["Enrollment", "Name", date_chosen]]

        # Function to apply styles to DataFrame
        def highlight_value(val):
            if val == 0:
                return "background-color: red"
            elif val == 1:
                return "background-color: green"
            else:
                return ""

        # Apply styles to DataFrame
        styled_df = date_df.style.applymap(highlight_value)

        ##########################
        ### DISPLAY STAT PLOTS ###
        ##########################

        #################################################
        ## Create subplots Absent / present comparison ##
        #################################################
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        absent_ct = date_df[date_df[date_chosen] == 0]["Name"].count()
        present_ct = date_df[date_df[date_chosen] == 1]["Name"].count()
        plt.bar(["Absent", "Present"], [absent_ct, present_ct], color=["red", "green"])
        ax1.set_xlabel("Attendance")
        ax1.set_ylabel("Count")
        ax1.set_title("Comparison of Absent and Present Students")
        plt.tight_layout()

        #######################################
        ## Attendance comparison gauge chart ##
        #######################################

        # Calculate the total number of enrolled students
        total_students = len(date_df)

        # Calculate the number of students present
        present_students = date_df[date_chosen].sum()

        # Calculate the percentage of present students
        percentage_present = (present_students / total_students) * 100

        # Create a gauge chart
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        # Plot the gauge
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Plot the colored arc representing the percentage present
        ax2.fill_between([0, percentage_present], 0, 100, color="green", alpha=0.6)

        # Plot the grey arc representing the remaining percentage
        ax2.fill_between([percentage_present, 100], 0, 100, color="red", alpha=0.6)

        # Add a text label indicating the percentage present
        ax2.text(
            50, 50, f"{percentage_present:.2f}%", ha="center", va="center", fontsize=20
        )
        plt.tight_layout()

        ## Show the data first in a container
        with st.container(border=True):
            st.subheader(f"Attendance data for {subject} 📊 on date {date_chosen} 🗓️")
            st.dataframe(styled_df, use_container_width=True)

        ####################
        ## Main container ##
        ####################
        with st.container(border=True):
            st.subheader(f"Attendance stat plots for subject '{subject}'")
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                # Container for Plot 1
                with st.container(border=True):
                    st.subheader("Absent / present comparison")
                    st.markdown(
                        "This plot shows the comparison between total students absent and present"
                    )
                    st.pyplot(fig1)

            with col2:
                # Container for Plot 2
                with st.container(border=True):
                    st.subheader("Attendance comparison gauge chart")
                    st.markdown(
                        "This plot shows the gauge chart for present students out of total students"
                    )
                    st.pyplot(fig2)


if __name__ == "__main__":
    main()
