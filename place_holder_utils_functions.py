
def draw_age_graph():
    """
    Plots the age distribution with respect to the default probability and the applicant's position
    
    Parameters :
    Return :
    """
    
    age_data = app_train[['TARGET', 'DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

    # Bin the age data
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    # Group by the bin and calculate averages
    age_groups  = age_data.groupby('YEARS_BINNED').mean()
    plt.figure(figsize = (8, 8))

    # Graph the age bins and the average of the target as a bar plot
    plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

    # Plot labeling
    plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by Age Group');
    
    return
    
def draw_salary_graph():
    """
    Plots the salary distribution with respect to the default probability and the applicant's position
    
    Parameters :
    Return :
    """
    return
    
def draw_credit_graph():
    """
    Plots the credit amount distribution with respect to the default probability and the applicant's position
    
    Parameters :
    Return :
    """
    return`
    
def feature_importance_lime():
    """
    Plots the feature importance of the model for the applicant
    
    Parameters :
    Return :
    """
    return




