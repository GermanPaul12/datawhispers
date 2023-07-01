import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import chi2_contingency, ttest_ind

# Korrelationstabellenerstellung #

def Correlation_table(df:pd.DataFrame, max:int=10):
    """
    [numerisch] - [numerisch]\n
    Stellt die stärksten Korrelationen zwischen intervallskalierten Variablen sortiert visuell dar.\n 
    Größeres p --> Größere Abhängigkeit
    @Variabel df: Zugehöriges DataFrame
    @Variabel max: Maximale Anzahl der Korrelationen
    """
    print("--------------------------------------------------")
    print("|               Korrelationstabelle              |")
    print("--------------------------------------------------")
    
    correlations=[]
    for var in df.dtypes.items():
        if var[1].name=="int64" or var[1].name=="float64":
            
            for othervar in df.dtypes.items():
                if (var[0]==othervar[0]):
                    break
                if ((othervar[1].name=="int64" or othervar[1]=="float64")):        
                    correlation=df[var[0]].corr(df[othervar[0]])
                    correlations.append([abs(correlation),correlation,var[0]+"-"+othervar[0]])
    correlations.sort(reverse=True)
    heights=[]
    labels=[]
    if max > len(correlations):
        max=len(correlations)
    for i in range(max):
        correlation=correlations[i]
        heights.append(correlation[0])
        labels.append(correlation[2])
        
    figure,axis=plt.subplots()
    plt.xticks(rotation=90)
    axis.bar(x=np.arange(len(heights)),height=heights,tick_label=labels,color="gray")
    axis.set_ylabel("Pearson Correlation p");
    plt.show()
    
    

def Correlation_analysis_all(df:pd.DataFrame):
    
    print("--------------------------------------------------")
    print("|               Korrelationsanalyse              |")
    print("--------------------------------------------------")
    
    for var in df.dtypes.items():
        if var[1].name=="int64" or var[1].name=="float64":
            
            for othervar in df.dtypes.items():
                if (var[0]==othervar[0]):
                    break
                if ((othervar[1].name=="int64" or othervar[1]=="float64")): 
                    correlation=df[var[0]].corr(df[othervar[0]])
                    print(u"Pearsons Rho \u03C1 = ",abs(correlation))
                    if abs(correlation)>0.2:
                        print("!!!!Statistische Abhängigkeit!!!!")
                    else:
                        print()
                    figure, axis = plt.subplots()
                    axis.scatter(df[var[0]],df[othervar[0]],s=3,color="gray")
                    axis.set_title(f"{othervar[0]} vs. {var[0]}")
                    axis.set_ylabel(othervar[0])
                    axis.set_xlabel(var[0])
                    plt.tight_layout()
                    plt.show()
# T-Test #

def T_Test(df:pd.DataFrame, Zielvariable, Zielvariable_Wert_gut, Zielvariable_Wert_schlecht):
    """
    [metrisch] - [kategorisch]\n
    Stellt die unterschiedlichen stat. Abhängigkeiten mit Mittelwert und Standardabweichung visuell dar
    @Zielvariable: Nur zwei Ausprägungen
    """
    
    print("--------------------------------------------------")
    print("|                     T-Test                     |")
    print("--------------------------------------------------")
    
    subset_good=df[df[Zielvariable]==Zielvariable_Wert_gut]
    subset_bad=df[df[Zielvariable]==Zielvariable_Wert_schlecht]
    for testvar in df.keys():
        if df.dtypes[testvar]=="int64" or df.dtypes[testvar]=="float64":
            tval,pval=ttest_ind(subset_good[testvar],subset_bad[testvar])
            print(testvar,"-",Zielvariable,":","\nt =",'{:.1f}'.format(abs(tval)),"\np =",'{:.1f}'.format(pval),pval)
            
            figure,axis=plt.subplots()
            heights=[subset_good[testvar].mean(),subset_bad[testvar].mean()]
            stds=[[0,0],[subset_good[testvar].std(),subset_bad[testvar].std()]]
            print(Zielvariable_Wert_gut,": ",u"\u03bc \u2248 ",'{:.4f}'.format(heights[0]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(stds[1][0]))
            print(Zielvariable_Wert_schlecht,": ",u"\u03bc \u2248 ",'{:.4f}'.format(heights[1]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(stds[1][1]))      
            axis.bar(x=np.arange(len(heights)),height=heights,tick_label=[Zielvariable_Wert_gut, Zielvariable_Wert_schlecht],color="gray",yerr=stds,ecolor="gray",width=0.4,capsize=4)
            axis.set_title(f"{testvar} vs. {Zielvariable}")
            axis.set_ylabel(f"Durchschnitt {testvar}");
            axis.set_xlabel(Zielvariable);
            plt.tight_layout()
            plt.show()

# chi2 - Test #

def chi_square_einzeln(df:pd.DataFrame, varname, Zielvariable):
    """
    [kategorisch] - [kategorisch]
    """
    contingencyTable=pd.crosstab(df[varname],df[Zielvariable])
    chi2,pval,dof,expectedFreq=chi2_contingency(contingencyTable)
    if pval<0.05:
        print(varname,"--",Zielvariable,":","chi2=",'{:.1f}'.format(chi2),"p=",'{:.5f}'.format(pval),"dof=",dof)#show pval in non-scienetific expression
        #print("expected values\n",expectedFreq)
        #print(contingencyTable)
    else:
        print(f"Es gibt keinen stat. Zusammenhang p>0.05 bei {varname} --> p={pval};chi2={chi2}")
    return contingencyTable

def chi_square_einzeln_mit_Erwartungswerten(df:pd.DataFrame, varname, Zielvariable):
    """
    [kategorisch] - [kategorisch]\n
    Mit Tabelle
    """
    contingencyTable=pd.crosstab(df[varname],df[Zielvariable])
    chi2,pval,dof,expectedFreq=chi2_contingency(contingencyTable)
    if pval<0.05:
        print(varname,"--",Zielvariable,":",u"\u03C7\u00B2=",'{:.1f}'.format(chi2),"p=",'{:.5f}'.format(pval),"dof=",dof)#show pval in non-scienetific expression
        print("expected values\n",expectedFreq)
        print(contingencyTable)
    else:
        print(f"Es gibt keinen stat. Zusammenhang p>0.05 bei {varname} --> p={pval} \u03C7\u00B2={chi2}")
    return contingencyTable

def chi_square_all(df:pd.DataFrame, Zielvariable):
    """
    [kategorisch] - [kategorisch]\n
    Erstellt alle möglichen chi2-Werte und p werte\n
    p<0.05 und chi2 sehr groß --> Signifikant
    """
    print("--------------------------------------------------")
    print("|                   Chi² - Test                  |")
    print("--------------------------------------------------")
    for var in df.dtypes.items():
        if var[1].name=="object" and var[0]!=Zielvariable:
            chi_square_einzeln_mit_Erwartungswerten(df, var[0], Zielvariable)
            print(30*"---")

def Analyse(df:pd.DataFrame,Zielvariable:str ,Zielvariable_gut,Zielvariable_schlecht,max=20):
    """
    Eine rundumanalyse eines Datensatzes in statistischer Form\n
    Beachte, dass die Zielvariable kategorisch sein sollte, wenn nicht benutzte .map um es einzufügen
    """
    print("                                   --------------------------------------------------")
    print("                                   |                     Analyse                    |")
    print("                                   --------------------------------------------------")
    Correlation_table(df,max) # Numerische Korrelationen
    #time.sleep(0.5)
    Correlation_analysis_all(df) # Grafiken zu numerischen Korrelationen
    T_Test(df,Zielvariable,Zielvariable_gut,Zielvariable_schlecht) # Numerisch - Kategorischen Korrelationen
    chi_square_all(df,Zielvariable) # Kategorisch - Kategorisch Korrelationen