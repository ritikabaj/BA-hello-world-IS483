import pandas as pd

def main():
    # Read the data
    df = pd.read_excel('./input/SMU_Survey_Data.xlsx')

    # Create Excel writer object for the consolidated file
    excel_filename = './output/SMU_Survey_Final.xlsx'
    excel_writer = pd.ExcelWriter(excel_filename, engine='openpyxl')

    # Write the original data as the first sheet named 'Data'
    df.to_excel(excel_writer, sheet_name='Data', index=False)
    print(f"Original data written to 'Data' sheet")

    def school_year_data():
        '''Process and write school_year_data Sheet'''
        # Cross-tabulation data for heatmaps
        # School vs Year of Study
        school_year_crosstab = pd.crosstab(df['School'], df['YearOfStudy'], margins=False)
        school_year_long = school_year_crosstab.reset_index().melt(id_vars='School', var_name='YearOfStudy', value_name='Count')

        # Write to Excel sheet
        school_year_long.to_excel(excel_writer, sheet_name='school_year_data', index=False)
        print("School year data written to 'school_year_data' sheet")

    def wellness_data():
        '''Process and write wellness_long_data and wellness_summary_data Sheets'''
        # Define wellness dimension columns and clean names
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

        # 1. Long format for radar charts, box plots, bar charts
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

        # Write long format wellness data to Excel sheet
        df_wellness_long.to_excel(excel_writer, sheet_name='wellness_long_data', index=False)
        print("Long format wellness data written to 'wellness_long_data' sheet")

        # 2. Summary statistics for heatmaps
        wellness_summary = df_wellness_long.groupby(['School', 'WellnessDimension'])['Score'].mean().reset_index()
        wellness_summary['Score'] = wellness_summary['Score'].round(2)

        # Write summary data to Excel sheet
        wellness_summary.to_excel(excel_writer, sheet_name='wellness_summary_data', index=False)
        print("Wellness summary data written to 'wellness_summary_data' sheet")

    def challenges_data():
        '''Process and write challenges_long_data Sheet'''
        # Create a long format where each challenge gets its own row
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

        # Write to Excel sheet
        df_challenges_long.to_excel(excel_writer, sheet_name='challenges_long_data', index=False)
        print("Long format challenges data written to 'challenges_long_data' sheet")

    def phase_data():
        '''Process and write phase_long_data Sheet'''
        # Create a long format for phase data
        phase_data = []
        phase_cols = [col for col in df.columns if 'Q6_' in col]

        # Clean up phase column names for better visualization
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
                phase_data.append({
                    'ResponseID': row['ResponseID'],
                    'Age': row['Age'],
                    'Gender': row['Gender'],
                    'YearOfStudy': row['YearOfStudy'],
                    'School': row['School'],
                    'Phase': phase_names[phase_col],
                    'MentalHealthRating': row[phase_col]
                })

        df_phases = pd.DataFrame(phase_data)

        # Write to Excel sheet
        df_phases.to_excel(excel_writer, sheet_name='phase_long_data', index=False)
        print("Phase data written to 'phase_long_data' sheet")

    def columnW_long():
        '''Process and write columnW_long_data Sheet'''
        col_w = 'Q9_9WhatMostMotivatesYouToTakeCareOfYourHealthSelectUpTo3'

        # Extract all unique motivations from the comma-separated responses
        all_motivations = set()
        for response in df[col_w].dropna():
            if isinstance(response, str):
                motivations = [m.strip() for m in response.split(',')]
                all_motivations.update(motivations)

        # Create long format for individual motivation analysis
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

        # Write to Excel sheet
        df_motivations_long.to_excel(excel_writer, sheet_name='ColumnW_long', index=False)
        print("ColumnW_long data written to sheet")

    def q11_q12_data():
        '''Process and write Q11_long, Q12_long, Q11_summary, Q12_summary Sheets'''
        # Define the service names for cleaner labels
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

        # AWARENESS DATA (Q11 - Columns Y to AF)
        cols_y_to_af = df.columns[25:33]  # Z to AG
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

        # USAGE DATA (Q12 - Columns AG to AN)
        cols_ag_to_an = df.columns[33:41]  # AH to AO  
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

        # Write long format files to Excel sheets
        df_awareness_long.to_excel(excel_writer, sheet_name='Q11_long', index=False)
        print("Awareness long format written to 'Q11_long' sheet")

        df_usage_long.to_excel(excel_writer, sheet_name='Q12_long', index=False)
        print("Usage long format written to 'Q12_long' sheet")

        # Awareness summary
        awareness_summary = df_awareness_long.groupby(['Service', 'Response']).size().reset_index(name='Count')
        awareness_summary['Percentage'] = awareness_summary.groupby('Service')['Count'].transform(lambda x: (x / x.sum() * 100).round(2))
        awareness_summary.to_excel(excel_writer, sheet_name='Q11_summary', index=False)
        print("Services summary written to 'Q11_summary' sheet")

        # Usage summary  
        usage_summary = df_usage_long.groupby(['Service', 'Response']).size().reset_index(name='Count')
        usage_summary['Percentage'] = usage_summary.groupby('Service')['Count'].transform(lambda x: (x / x.sum() * 100).round(2))
        usage_summary.to_excel(excel_writer, sheet_name='Q12_summary', index=False)
        print("Usage summary written to 'Q12_summary' sheet")

    def q13_q14_data():
        '''Process and write Q13_long, Q14_long Sheets'''
        # Column AP is index 41, BA is index 52
        cols_ao_to_az = df.columns[41:53]  # AP to BA (12 columns)

        # Define clean names for the health habits
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

        # Split columns into Q13 and Q14 groups
        q13_cols = [col for col in cols_ao_to_az if col.startswith('Q13_')]
        q14_cols = [col for col in cols_ao_to_az if col.startswith('Q14_')]

        # Q13 CURRENT HEALTH HABITS (Long Format)
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

        # Q14 DESIRED HEALTH HABITS (Long Format)
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

        # Write to Excel sheets
        df_q13_long.to_excel(excel_writer, sheet_name='Q13_long', index=False)
        print("✅ Q13 - Current habits written to 'Q13_long' sheet")

        df_q14_long.to_excel(excel_writer, sheet_name='Q14_long', index=False)
        print("✅ Q14 - Desired habits written to 'Q14_long' sheet")

    def q16_data():
        '''Process and write Q16_long Sheet'''
        # Use the actual demographic columns found
        column_bb_name = 'Q16_16IfYesWhichHabitSChanged'

        # Create long format data with all demographics
        bb_complete_long_data = []

        for idx, row in df.iterrows():
            # Get BB response (habits)
            bb_response = row[column_bb_name]
            
            # Base demographic data for this student
            base_data = {
                'ResponseID': row['ResponseID'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'YearOfStudy': row['YearOfStudy'],
                'School': row['School']
            }
            
            if pd.notna(bb_response) and str(bb_response).strip():
                # Split comma-separated habits
                habits = [habit.strip() for habit in str(bb_response).split(',')]
                
                # Create a row for each habit
                for habit in habits:
                    if habit:  # Skip empty habits
                        row_data = base_data.copy()
                        row_data['Habit'] = habit
                        bb_complete_long_data.append(row_data)
            else:
                # Empty response - create one row with NIL
                row_data = base_data.copy()
                row_data['Habit'] = 'NIL'
                bb_complete_long_data.append(row_data)

        # Convert to DataFrame
        df_bb_complete_long = pd.DataFrame(bb_complete_long_data)

        # Write to Excel sheet
        df_bb_complete_long.to_excel(excel_writer, sheet_name='Q16_long', index=False)
        print("Q16 data written to 'Q16_long' sheet")

    def q18_q19_q20_q21_data():
        '''Process and write Q18_long, Q19_long, Q20_long, Q21_long Sheets'''
        # Define the columns to process
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

        # Function to create long format with demographics
        def create_demo_long_format(column_name):
            """Create long format data with demographics for a given column"""
            long_data = []
            
            for _, row in df.iterrows():
                # Get demographic data for this student
                base_data = {
                    'ResponseID': row['ResponseID'],
                    'Age': row['Age'],
                    'Gender': row['Gender'],
                    'YearOfStudy': row['YearOfStudy'],
                    'School': row['School']
                }
                
                # Get column response
                response = row[column_name]
                
                if pd.notna(response) and str(response).strip():
                    # Split comma-separated responses
                    items = [item.strip() for item in str(response).split(',')]
                    
                    # Create a row for each item
                    for item in items:
                        if item:  # Skip empty items
                            row_data = base_data.copy()
                            row_data['Response'] = item
                            long_data.append(row_data)
                else:
                    # Empty response - create one row with NIL
                    row_data = base_data.copy()
                    row_data['Response'] = 'NIL'
                    long_data.append(row_data)
            
            return pd.DataFrame(long_data)

        # Process each column
        dataframes = {}

        for col_question, col_info in questions_to_process.items():
            # Create long format
            df_long = create_demo_long_format(col_info['name'])
            dataframes[col_question] = df_long

        # Write each column to its own Excel sheet
        for col_question, df_data in dataframes.items():
            # Write to Excel sheet
            df_data.to_excel(excel_writer, sheet_name=f'{col_question}_long', index=False)
            print(f"✅ Sheet '{col_question}_long' written with {len(df_data)} rows")

    def q32_data():
        '''Process and write Q32_long Sheet'''
        # Column BR: Q32_32WhichTimesWorkBestForYouSelectUpTo2 (multi-response, comma-separated)
        br_data = []

        for index, row in df.iterrows():
            response_id = row['ResponseID']
            age = row['Age']
            gender = row['Gender']
            year_of_study = row['YearOfStudy']
            school = row['School']
            
            br_value = row['Q32_32WhichTimesWorkBestForYouSelectUpTo2']
            
            if pd.isna(br_value) or br_value == '' or str(br_value).strip() == '':
                # If empty, add one row with NIL
                br_data.append({
                    'ResponseID': response_id,
                    'Age': age,
                    'Gender': gender,
                    'YearOfStudy': year_of_study,
                    'School': school,
                    'Response': 'NIL',
                })
            else:
                # Split by comma and create separate rows
                responses = [resp.strip() for resp in str(br_value).split(',')]
                for response in responses:
                    if response:  # Only add non-empty responses
                        br_data.append({
                            'ResponseID': response_id,
                            'Age': age,
                            'Gender': gender,
                            'YearOfStudy': year_of_study,
                            'School': school,
                            'Response': response,
                        })

        # Create DataFrame
        br_df = pd.DataFrame(br_data)

        # Write to Excel sheet
        br_df.to_excel(excel_writer, sheet_name='Q32_long', index=False)
        print("Q32 data written to 'Q32_long' sheet")

    def q33_q34_q35_data():
        '''Process and write Q33_long, Q34_long, Q35_long Sheets'''
        # Define question groups and their column mappings
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

        # Store combined dataframes
        enhanced_combined_dfs = {}

        for question_group, group_info in enhanced_question_groups.items():
            # Combine all data for this question group
            enhanced_data = []
            
            for col_letter in group_info['columns']:
                nice_question_name = group_info['question_names'][col_letter]
                original_column_name = question_groups[question_group]['question_names'][col_letter]
                
                # Get data for this column from our individual transforms
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
                        'Question': nice_question_name,  # Using the nice name
                        'Response': response_value,
                    })
            
            # Create combined DataFrame for this question group
            enhanced_df = pd.DataFrame(enhanced_data)

            # Sort by ResponseID and Question for consistent ordering
            enhanced_df_sorted = enhanced_df.sort_values(['ResponseID', 'Question'], ascending=[True, True])
            enhanced_combined_dfs[question_group] = enhanced_df_sorted

        # Write combined data for each question group to Excel sheets
        for question_group, combined_df in enhanced_combined_dfs.items():
            # Sort by ResponseID first, then by Question to ensure consistent order within each respondent
            sorted_df = combined_df.sort_values(['ResponseID', 'Question'], ascending=[True, True])
            
            # Write to Excel sheet
            sorted_df.to_excel(excel_writer, sheet_name=f'{question_group}_long', index=False)
            print(f"Combined data for {question_group} written to '{question_group}_long' sheet")

    def q36_data():
        '''Process and write Q36_long Sheet'''
        # Column CL: Q36_36WhichAreasOfTheResilienceFrameworkDoYouThinkSmuShouldProvideMoreInitiativesOrSupportIn (multi-response, comma-separated)
        cl_data = []

        for index, row in df.iterrows():
            response_id = row['ResponseID']
            age = row['Age']
            gender = row['Gender']
            year_of_study = row['YearOfStudy']
            school = row['School']
            
            cl_value = row['Q36_36WhichAreasOfTheResilienceFrameworkDoYouThinkSmuShouldProvideMoreInitiativesOrSupportIn']
            
            if pd.isna(cl_value) or cl_value == '' or str(cl_value).strip() == '':
                # If empty, add one row with NIL
                cl_data.append({
                    'ResponseID': response_id,
                    'Age': age,
                    'Gender': gender,
                    'YearOfStudy': year_of_study,
                    'School': school,
                    'Response': 'NIL',
                })
            else:
                # Split by comma and create separate rows
                responses = [resp.strip() for resp in str(cl_value).split(',')]
                for response in responses:
                    if response:  # Only add non-empty responses
                        cl_data.append({
                            'ResponseID': response_id,
                            'Age': age,
                            'Gender': gender,
                            'YearOfStudy': year_of_study,
                            'School': school,
                            'Response': response
                        })

        # Create DataFrame
        cl_df = pd.DataFrame(cl_data)

        # Write to Excel sheet
        cl_df.to_excel(excel_writer, sheet_name='Q36_long', index=False)
        print("Q36 data written to 'Q36_long' sheet")


    # Call all processing functions
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

    # Close the Excel writer to save the file
    excel_writer.close()
    print(f"\n🎉 All data successfully written to '{excel_filename}'")
    print("This file contains multiple sheets with all the processed data.")

if __name__ == "__main__":
    main()