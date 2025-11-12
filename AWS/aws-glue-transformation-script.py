import sys
import boto3
import pandas as pd
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

s3 = boto3.client('s3')

# Your bucket and object keys
input_bucket = 'fyp-hello-world'
input_key = 'input-folder/SMU_Survey_Data.xlsx'
output_bucket = 'fyp-hello-world'
output_key = 'output-folder/SMU_Survey_Final.xlsx'

# Local temp paths for Glue job container
local_input_path = '/tmp/input.xlsx'
local_output_path = '/tmp/output.xlsx'

# Download input Excel from S3
s3.download_file(input_bucket, input_key, local_input_path)

# Load Excel into pandas DataFrame
df = pd.read_excel(local_input_path)

# Prepare the Excel writer
excel_writer = pd.ExcelWriter(local_output_path, engine='openpyxl')

# Write original data sheet
df.to_excel(excel_writer, sheet_name='Data', index=False)
print(f"Original data written to 'Data' sheet")

# (Include all your functions below unmodified, except remove file paths and prints can stay for logs)

def school_year_data():
    school_year_crosstab = pd.crosstab(df['School'], df['YearOfStudy'], margins=False)
    school_year_long = school_year_crosstab.reset_index().melt(id_vars='School', var_name='YearOfStudy', value_name='Count')
    school_year_long.to_excel(excel_writer, sheet_name='school_year_data', index=False)
    print("School year data written to 'school_year_data' sheet")

def wellness_data():
    wellness_cols = ['Q4_stressLevels', 'Q4_sleepQuality', 'Q4_physicalHealth', 
                    'Q4_academicPressure', 'Q4_socialConnectedness', 'Q4_overallMentalHealth']
    wellness_names = {
        'Q4_stressLevels': 'Stress Levels',
        'Q4_sleepQuality': 'Sleep Quality', 
        'Q4_physicalHealth': 'Physical Health',
        'Q4_academicPressure': 'Academic Pressure',
        'Q4_socialConnectedness': 'Social Connectedness',
        'Q4_overallMentalHealth': 'Overall Mental Health'
    }
    wellness_long = []
    for _, row in df.iterrows():
        for col in wellness_cols:
            wellness_long.append({
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School'],
                'WellnessDimension': wellness_names[col],
                'Score': row[col]
            })
    df_wellness_long = pd.DataFrame(wellness_long)
    df_wellness_long.to_excel(excel_writer, sheet_name='wellness_long_data', index=False)
    print("Long format wellness data written to 'wellness_long_data' sheet")

    wellness_summary = df_wellness_long.groupby(['School', 'WellnessDimension'])['Score'].mean().reset_index()
    wellness_summary['Score'] = wellness_summary['Score'].round(2)
    wellness_summary.to_excel(excel_writer, sheet_name='wellness_summary_data', index=False)
    print("Wellness summary data written to 'wellness_summary_data' sheet")

def challenges_data():
    challenges_long = []
    for _, row in df.iterrows():
        if pd.notna(row['Q5_5WhatAreTheMainChallengesAffectingYourMentalHealthSelectUpTo3']):
            response_challenges = [c.strip() for c in row['Q5_5WhatAreTheMainChallengesAffectingYourMentalHealthSelectUpTo3'].split(',')]
            for challenge in response_challenges:
                challenges_long.append({
                    'ResponseID': row['ResponseID'],
                    'Age': row['Age'],
                    'Gender': row['Gender'],
                    'YearOfStudy': row['YearOfStudy'],
                    'School': row['School'],
                    'Challenge': challenge
                })
    df_challenges_long = pd.DataFrame(challenges_long)
    df_challenges_long.to_excel(excel_writer, sheet_name='challenges_long_data', index=False)
    print("Long format challenges data written to 'challenges_long_data' sheet")

def phase_data():
    phase_data_arr = []
    phase_cols = [col for col in df.columns if 'Q6_' in col]
    phase_names = {
        'Q6_preSemesterBeforeClassesStarted': 'Pre-Semester',
        'Q6_startOfSemesterWeeks14': 'Start of Semester (Weeks 1-4)',
        'Q6_midSemesterWeeks57ProjectAssignmentPeriod': 'Mid-Semester (Weeks 5-7)',
        'Q6_recessWeekWeek8': 'Recess Week (Week 8)',
        'Q6_endOfSemesterWeeks912FinalProjectAssignmentPeriod': 'End of Semester (Weeks 9-12)',
        'Q6_finalExamPeriod': 'Final Exam Period',
        'Q6_postSemesterAfterExamsEnded': 'Post-Semester'
    }
    for _, row in df.iterrows():
        for phase_col in phase_cols:
            phase_data_arr.append({
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School'],
                'Phase': phase_names[phase_col],
                'MentalHealthRating': row[phase_col]
            })
    df_phases = pd.DataFrame(phase_data_arr)
    df_phases.to_excel(excel_writer, sheet_name='phase_long_data', index=False)
    print("Phase data written to 'phase_long_data' sheet")

def columnW_long():
    col_w = 'Q9_9WhatMostMotivatesYouToTakeCareOfYourHealthSelectUpTo3'
    all_motivations = set()
    for response in df[col_w].dropna():
        if isinstance(response, str):
            motivations = [m.strip() for m in response.split(',')]
            all_motivations.update(motivations)
    motivations_long = []
    for _, row in df.iterrows():
        if isinstance(row[col_w], str):
            motivations = [m.strip() for m in row[col_w].split(',')]
            for motivation in motivations:
                motivations_long.append({
                    'ResponseID': row['ResponseID'],
                    'Age': row['Age'],
                    'Gender': row['Gender'],
                    'YearOfStudy': row['YearOfStudy'],
                    'School': row['School'],
                    'Motivation': motivation,
                    'Selected': 1
                })
    df_motivations_long = pd.DataFrame(motivations_long)
    df_motivations_long.to_excel(excel_writer, sheet_name='ColumnW_long', index=False)
    print("ColumnW_long data written to sheet")

def q11_q12_data():
    service_names = {
        'Q11_mentalHealthWeek': 'Mental Health Week',
        'Q11_resilienceFramework': 'Resilience Framework',
        'Q11_examAngels': 'Exam Angels',
        'Q11_studentCareServices': 'Student Care Services',
        'Q11_cosyHaven': 'Cosy Haven',
        'Q11_voicesRoadshows': 'Voices Roadshows',
        'Q11_peerHelpersRoadshows': 'Peer Helpers Roadshows',
        'Q11_careerCompassByStudentWellnessCentre': 'Career Compass',
        'Q12_mentalHealthWeek': 'Mental Health Week',
        'Q12_resilienceFramework': 'Resilience Framework',
        'Q12_examAngels': 'Exam Angels',
        'Q12_studentCareServices': 'Student Care Services',
        'Q12_cosyHaven': 'Cosy Haven',
        'Q12_voicesRoadshows': 'Voices Roadshows',
        'Q12_peerHelpersRoadshows': 'Peer Helpers Roadshows',
        'Q12_careerCompassByStudentWellnessCentre': 'Career Compass'
    }
    cols_y_to_af = df.columns[25:33]
    awareness_cols = list(cols_y_to_af)
    awareness_long = []
    for _, row in df.iterrows():
        for col in awareness_cols:
            service_name = service_names[col]
            awareness_long.append({
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School'],
                'Service': service_name,
                'Response': row[col]
            })
    df_awareness_long = pd.DataFrame(awareness_long)
    cols_ag_to_an = df.columns[33:41]
    usage_cols = list(cols_ag_to_an)
    usage_long = []
    for _, row in df.iterrows():
        for col in usage_cols:
            service_name = service_names[col]
            usage_long.append({
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School'],
                'Service': service_name,
                'Response': row[col]
            })
    df_usage_long = pd.DataFrame(usage_long)
    df_awareness_long.to_excel(excel_writer, sheet_name='Q11_long', index=False)
    print("Awareness long format written to 'Q11_long' sheet")
    df_usage_long.to_excel(excel_writer, sheet_name='Q12_long', index=False)
    print("Usage long format written to 'Q12_long' sheet")
    awareness_summary = df_awareness_long.groupby(['Service', 'Response']).size().reset_index(name='Count')
    awareness_summary['Percentage'] = awareness_summary.groupby('Service')['Count'].transform(lambda x: (x / x.sum() * 100).round(2))
    awareness_summary.to_excel(excel_writer, sheet_name='Q11_summary', index=False)
    print("Services summary written to 'Q11_summary' sheet")
    usage_summary = df_usage_long.groupby(['Service', 'Response']).size().reset_index(name='Count')
    usage_summary['Percentage'] = usage_summary.groupby('Service')['Count'].transform(lambda x: (x / x.sum() * 100).round(2))
    usage_summary.to_excel(excel_writer, sheet_name='Q12_summary', index=False)
    print("Usage summary written to 'Q12_summary' sheet")

def q13_q14_data():
    cols_ao_to_az = df.columns[41:53]
    habit_names = {
        'Q13_studyLifeBalance': 'Study-Life Balance',
        'Q13_exerciseHabits': 'Exercise Habits',
        'Q13_stressCopingHabitsEGDrinkingGamingSocialisingMindfulness': 'Stress Coping Habits',
        'Q13_eatingHabits': 'Eating Habits',
        'Q13_timeManagement': 'Time Management',
        'Q13_sleepHabits': 'Sleep Habits',
        'Q14_studyLifeBalance': 'Study-Life Balance',
        'Q14_exerciseHabits': 'Exercise Habits',
        'Q14_stressCopingHabitsEGDrinkingGamingSocialisingMindfulness': 'Stress Coping Habits',
        'Q14_eatingHabits': 'Eating Habits',
        'Q14_timeManagement': 'Time Management',
        'Q14_sleepHabits': 'Sleep Habits'
    }
    q13_cols = [col for col in cols_ao_to_az if col.startswith('Q13_')]
    q14_cols = [col for col in cols_ao_to_az if col.startswith('Q14_')]
    q13_long = []
    for _, row in df.iterrows():
        for col in q13_cols:
            habit_name = habit_names[col]
            q13_long.append({
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School'],
                'HealthHabit': habit_name,
                'Rating': row[col],
            })
    df_q13_long = pd.DataFrame(q13_long)
    q14_long = []
    for _, row in df.iterrows():
        for col in q14_cols:
            habit_name = habit_names[col]
            q14_long.append({
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School'],
                'HealthHabit': habit_name,
                'Rating': row[col],
            })
    df_q14_long = pd.DataFrame(q14_long)
    df_q13_long.to_excel(excel_writer, sheet_name='Q13_long', index=False)
    print("✅ Q13 - Current habits written to 'Q13_long' sheet")
    df_q14_long.to_excel(excel_writer, sheet_name='Q14_long', index=False)
    print("✅ Q14 - Desired habits written to 'Q14_long' sheet")

def q16_data():
    column_bb_name = 'Q16_16IfYesWhichHabitSChanged'
    bb_complete_long_data = []
    for idx, row in df.iterrows():
        bb_response = row[column_bb_name]
        base_data = {
            'ResponseID': row['ResponseID'],
            'Age': row['Age'],
            'Gender': row['Gender'],
            'YearOfStudy': row['YearOfStudy'],
            'School': row['School']
        }
        if pd.notna(bb_response) and str(bb_response).strip():
            habits = [habit.strip() for habit in str(bb_response).split(',')]
            for habit in habits:
                if habit:
                    row_data = base_data.copy()
                    row_data['Habit'] = habit
                    bb_complete_long_data.append(row_data)
        else:
            row_data = base_data.copy()
            row_data['Habit'] = 'NIL'
            bb_complete_long_data.append(row_data)
    df_bb_complete_long = pd.DataFrame(bb_complete_long_data)
    df_bb_complete_long.to_excel(excel_writer, sheet_name='Q16_long', index=False)
    print("Q16 data written to 'Q16_long' sheet")

def q18_q19_q20_q21_data():
    questions_to_process = {
        'Q18': {
            'name': 'Q18_18IfYesWhichHabitSChanged',
            'question': 'Q18: Which Habits Changed'
        },
        'Q19': {
            'name': 'Q19_19WhichTopicsDoYouDiscussMostInTheLast3MonthsWithPeersFamilyEtcSelectUpTo3',
            'question': 'Q19: Topics Discussed Most'
        },
        'Q20': {
            'name': 'Q20_20WhereDoTheseConversationsUsuallyHappen',
            'question': 'Q20: Conversation Locations'
        },
        'Q21': {
            'name': 'Q21_21WhenDoYouUsuallyTalkMoreAboutMentalHealth',
            'question': 'Q21: Mental Health Discussion Timing'
        }
    }
    def create_demo_long_format(column_name):
        long_data = []
        for _, row in df.iterrows():
            base_data = {
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School']
            }
            response = row[column_name]
            if pd.notna(response) and str(response).strip():
                items = [item.strip() for item in str(response).split(',')]
                for item in items:
                    if item:
                        row_data = base_data.copy()
                        row_data['Response'] = item
                        long_data.append(row_data)
            else:
                row_data = base_data.copy()
                row_data['Response'] = 'NIL'
                long_data.append(row_data)
        return pd.DataFrame(long_data)

    dataframes = {}

    for col_question, col_info in questions_to_process.items():
        df_long = create_demo_long_format(col_info['name'])
        dataframes[col_question] = df_long

    for col_question, df_data in dataframes.items():
        df_data.to_excel(excel_writer, sheet_name=f'{col_question}_long', index=False)
        print(f"✅ Sheet '{col_question}_long' written with {len(df_data)} rows")

def q32_data():
    br_data = []
    for index, row in df.iterrows():
        response_id = row['ResponseID']
        age = row['Age']
        gender = row['Gender']
        year_of_study = row['YearOfStudy']
        school = row['School']
        br_value = row['Q32_32WhichTimesWorkBestForYouSelectUpTo2']
        if pd.isna(br_value) or br_value == '' or str(br_value).strip() == '':
            br_data.append({
                'ResponseID': response_id,
                'Age': age,
                'Gender': gender,
                'YearOfStudy': year_of_study,
                'School': school,
                'Response': 'NIL',
            })
        else:
            responses = [resp.strip() for resp in str(br_value).split(',')]
            for response in responses:
                if response:
                    br_data.append({
                        'ResponseID': response_id,
                        'Age': age,
                        'Gender': gender,
                        'YearOfStudy': year_of_study,
                        'School': school,
                        'Response': response,
                    })
    br_df = pd.DataFrame(br_data)
    br_df.to_excel(excel_writer, sheet_name='Q32_long', index=False)
    print("Q32 data written to 'Q32_long' sheet")

def q33_q34_q35_data():
    question_groups = {
        'Q33': {
            'columns': ['BS', 'BT', 'BU', 'BV'],
            'description': 'Wellness Barriers',
            'question_names': {
                'BS': 'Q33_lackOfTime',
                'BT': 'Q33_cost', 
                'BU': 'Q33_lackOfMotivationWillpower',
                'BV': 'Q33_convenience'
            }
        },
        'Q34': {
            'columns': ['BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE'],
            'description': 'SMU Wellness Program Awareness',
            'question_names': {
                'BW': 'Q34_mentalHealthWeek',
                'BX': 'Q34_resilienceFramework',
                'BY': 'Q34_examAngels',
                'BZ': 'Q34_studentCareServices',
                'CA': 'Q34_cosyHaven',
                'CB': 'Q34_voicesRoadshows',
                'CC': 'Q34_peerHelpersRoadshows',
                'CD': 'Q34_careerCompass',
                'CE': 'Q34_caresCorner'
            }
        },
        'Q35': {
            'columns': ['CF', 'CG', 'CH', 'CI', 'CJ', 'CK'],
            'description': 'Resilience Framework Ratings',
            'question_names': {
                'CF': 'Q35_physical',
                'CG': 'Q35_intellectual',
                'CH': 'Q35_emotional',
                'CI': 'Q35_social',
                'CJ': 'Q35_career',
                'CK': 'Q35_financial'
            }
        }
    }

    enhanced_question_groups = {
        'Q33': {
            'columns': ['BS', 'BT', 'BU', 'BV'],
            'description': 'Wellness Barriers',
            'question_names': {
                'BS': 'Lack of Time',
                'BT': 'Cost', 
                'BU': 'Lack of Motivation/Willpower',
                'BV': 'Convenience'
            }
        },
        'Q34': {
            'columns': ['BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE'],
            'description': 'SMU Wellness Program Awareness',
            'question_names': {
                'BW': 'Mental Health Week',
                'BX': 'Resilience Framework',
                'BY': 'Exam Angels',
                'BZ': 'Student Care Services',
                'CA': 'Cosy Haven',
                'CB': 'Voices Roadshows',
                'CC': 'Peer Helpers Roadshows',
                'CD': 'Career Compass',
                'CE': 'Cares Corner'
            }
        },
        'Q35': {
            'columns': ['CF', 'CG', 'CH', 'CI', 'CJ', 'CK'],
            'description': 'Resilience Framework Ratings',
            'question_names': {
                'CF': 'Physical Wellness',
                'CG': 'Intellectual Wellness',
                'CH': 'Emotional Wellness',
                'CI': 'Social Wellness',
                'CJ': 'Career Wellness',
                'CK': 'Financial Wellness'
            }
        }
    }
    enhanced_combined_dfs = {}
    for question_group, group_info in enhanced_question_groups.items():
        enhanced_data = []
        for col_letter in group_info['columns']:
            nice_question_name = group_info['question_names'][col_letter]
            original_column_name = question_groups[question_group]['question_names'][col_letter]
            for index, row in df.iterrows():
                response_id = row['ResponseID']
                age = row['Age']
                gender = row['Gender']
                year_of_study = row['YearOfStudy']
                school = row['School']
                col_value = row[original_column_name]
                if pd.isna(col_value) or col_value == '' or str(col_value).strip() == '':
                    response_value = 'NIL'
                else:
                    response_value = str(col_value)
                enhanced_data.append({
                    'ResponseID': response_id,
                    'Age': age,
                    'Gender': gender,
                    'YearOfStudy': year_of_study,
                    'School': school,
                    'Question': nice_question_name,
                    'Response': response_value,
                })
        enhanced_df = pd.DataFrame(enhanced_data)
        enhanced_df_sorted = enhanced_df.sort_values(['ResponseID', 'Question'], ascending=[True, True])
        enhanced_combined_dfs[question_group] = enhanced_df_sorted

    for question_group, combined_df in enhanced_combined_dfs.items():
        sorted_df = combined_df.sort_values(['ResponseID', 'Question'], ascending=[True, True])
        sorted_df.to_excel(excel_writer, sheet_name=f'{question_group}_long', index=False)
        print(f"Combined data for {question_group} written to '{question_group}_long' sheet")

def q36_data():
    cl_data = []
    for index, row in df.iterrows():
        response_id = row['ResponseID']
        age = row['Age']
        gender = row['Gender']
        year_of_study = row['YearOfStudy']
        school = row['School']
        cl_value = row['Q36_36WhichAreasOfTheResilienceFrameworkDoYouThinkSmuShouldProvideMoreInitiativesOrSupportIn']
        if pd.isna(cl_value) or cl_value == '' or str(cl_value).strip() == '':
            cl_data.append({
                'ResponseID': response_id,
                'Age': age,
                'Gender': gender,
                'YearOfStudy': year_of_study,
                'School': school,
                'Response': 'NIL',
            })
        else:
            responses = [resp.strip() for resp in str(cl_value).split(',')]
            for response in responses:
                if response:
                    cl_data.append({
                        'ResponseID': response_id,
                        'Age': age,
                        'Gender': gender,
                        'YearOfStudy': year_of_study,
                        'School': school,
                        'Response': response
                    })
    cl_df = pd.DataFrame(cl_data)
    cl_df.to_excel(excel_writer, sheet_name='Q36_long', index=False)
    print("Q36 data written to 'Q36_long' sheet")

# Call all functions
school_year_data()
wellness_data()
challenges_data()
phase_data()
columnW_long()
q11_q12_data()
q13_q14_data()
q16_data()
q18_q19_q20_q21_data()
q32_data()
q33_q34_q35_data()
q36_data()

# Close Excel writer
excel_writer.close()
print(f"\n🎉 All data successfully written to '{local_output_path}'")

# Upload output Excel back to S3
s3.upload_file(local_output_path, output_bucket, output_key)

job.commit()
