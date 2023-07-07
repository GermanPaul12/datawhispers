import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from scipy.stats import chi2_contingency, ttest_ind, pearsonr

def Datensatz_bereinigen(df:pd.DataFrame,kategorische_Spalten:list or tuple=[],numerische_Spalten:list or tuple=[]):
    """
    Wandelt 'Numerische Daten' in Kategorische und anders herum.
    @return Bereinigter Datensatz als pd.DataFrame
    """
    for x in kategorische_Spalten:
        df[x]=df[x].apply(str)
    for x in numerische_Spalten:
        df[x]=df[x].apply(eval)
    return df

def T_Test_Rückgabe(df:pd.DataFrame, Zielvariable, Zielvariable_Wert_gut, Zielvariable_Wert_schlecht,Numerische_Grenze,Grafik=False):
    """
    [metrisch] - [kategorisch]\n
    Stellt die unterschiedlichen stat. Abhängigkeiten mit Mittelwert und Standardabweichung visuell dar
    @Zielvariable: Nur zwei Ausprägungen
    @return [Variable,p-Wert,t-wert,mü-gut, sigma-gut, mü-schlecht, sigma-schlecht]
    """
    P_Wert = []
    if df.dtypes[Zielvariable]=="int64" or df.dtypes[Zielvariable]=="float64":
        if Numerische_Grenze==0:
            Durchschnitt = df[Zielvariable].values.mean()
        else:
            Durchschnitt = Numerische_Grenze
        subset_good=df[df[Zielvariable]>Durchschnitt]
        subset_bad=df[df[Zielvariable]<Durchschnitt]
        Zielvariable_Wert_gut=f">{round(Durchschnitt,2)}"
        Zielvariable_Wert_schlecht=f"<{round(Durchschnitt,2)}"
    else:
        subset_good=df[df[Zielvariable]==Zielvariable_Wert_gut]
        subset_bad=df[df[Zielvariable]==Zielvariable_Wert_schlecht]
    for testvar in df.keys():
        if df.dtypes[testvar]=="int64" or df.dtypes[testvar]=="float64":
            tval,pval=ttest_ind(subset_good[testvar],subset_bad[testvar])
            
            heights=[subset_good[testvar].mean(),subset_bad[testvar].mean()]
            stds=[[0,0],[subset_good[testvar].std(),subset_bad[testvar].std()]]
            
            P_Wert.append((testvar,pval,tval,heights[0],stds[1][0],heights[1],stds[1][1],"","T-Test"))
    if Grafik:
        figure,axis=plt.subplots()
        print(Zielvariable_Wert_gut,": ",u"\u03bc \u2248 ",'{:.4f}'.format(heights[0]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(stds[1][0]))
        print(Zielvariable_Wert_schlecht,": ",u"\u03bc \u2248 ",'{:.4f}'.format(heights[1]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(stds[1][1]))      
        axis.bar(x=np.arange(len(heights)),height=heights,tick_label=[Zielvariable_Wert_gut, Zielvariable_Wert_schlecht],color="gray",yerr=stds,ecolor="gray",width=0.4,capsize=4)
        axis.set_title(f"{testvar} vs. {Zielvariable}")
        axis.set_ylabel(f"Durchschnitt {testvar}");
        axis.set_xlabel(Zielvariable);
        plt.tight_layout()
        plt.show()
    return P_Wert

def chi_square_all_Rückgabe(df:pd.DataFrame, Zielvariable,Numerische_Grenze):
    """
    [kategorisch] - [kategorisch]\n
    Erstellt alle möglichen chi2-Werte und p werte\n
    p<0.05 und chi2 sehr groß --> Signifikant
    """
    Ergebnis = []
    Tabelle = []
    for var in df.dtypes.items():
        if var[1].name=="object" and var[0]!=Zielvariable:
            Teillösung,expectedFreq,contigenzTable = chi_square_einzeln_Rückgabe(df, var[0], Zielvariable,Numerische_Grenze=Numerische_Grenze)
            Ergebnis.append(Teillösung)
            Tabelle.append((contigenzTable,expectedFreq))
    return Ergebnis,Tabelle

def chi_square_einzeln_Rückgabe(df:pd.DataFrame, varname, Zielvariable,Numerische_Grenze=0):
    """
    [kategorisch] - [kategorisch]
    """
    if df.dtypes[Zielvariable]=="int64" or df.dtypes[Zielvariable]=="float64":
        Tab = df[Zielvariable]
        if Numerische_Grenze==0:
            Durchschnitt = df[Zielvariable].values.mean()
            Tab = Tab>=Durchschnitt
        else:
            Durchschnitt = Numerische_Grenze
            Tab = Tab>=Durchschnitt        
        contingencyTable=pd.crosstab(df[varname],Tab)
        chi2,pval,dof,expectedFreq=chi2_contingency(contingencyTable)
        P_Wert=(varname,pval,chi2,"","","","",dof,"X²-Test")
    else:
        contingencyTable=pd.crosstab(df[varname],df[Zielvariable])
        chi2,pval,dof,expectedFreq=chi2_contingency(contingencyTable)
        P_Wert=(varname,pval,chi2,"","","","",dof,"X²-Test")
    return P_Wert,expectedFreq,contingencyTable

def Cor_tab(df,max=10):
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
    plt.title("Korrelationstabelle")
    plt.xticks(rotation=90)
    axis.bar(x=np.arange(len(heights)),height=heights,tick_label=labels,color="gray")
    axis.set_ylabel("Pearson Correlation p");
    plt.show()
            
def Analyse_Tabelle(df:pd.DataFrame,Zielvariable:str,Anzahl_an_gewünschten_Variablen:int=None,Zielvariable_good="",Zielvariable_bad="",Zielvariable_Grenze:int or float=0,kategorische_Variablen=[],Numerische_Variablen=[]):
    """
    Gibt die Tabelle mit den abhängigsten Variablen aus.
    @param df: Der DataFrame
    @param Zielvariable: Zielvariable
    @param Anzahl_an_gewünschten_Variablen (opt.)
    @param Zielvariable_good: Nur bei KATEGORISCHER Zielvariable
    @param Zielvariable_bad: Nur bei KATEGORISCHER Zielvariable
    @param Zielvariable_Grenze: Nur bei NUMERISCHER Zielvariable ;Schwellwert ab wann Gut oder Schlecht sonst Durchschnitt
    @param kategorische_Variablen: Eingeben einer Liste an kategorischen Variablen, die im Datensatz als numerisch gekennzeichnet sind (opt.) 
    @param numerische_Variablen: Eingeben einer Liste an numerischen Variablen, die im Datensatz als kategorisch gekennzeichnet sind (opt.) 
    @return Tabelle mit Variablen, Kennwerten, Testverfahren
    """
    ## Korregieren der Tabelle ##
        
    df = Datensatz_bereinigen(df,kategorische_Variablen,Numerische_Variablen)
    
    ## Erstellen der Variablentabelle sortiert nach größter Abhängigkeit
    
    # Tests
    
    T_Ergebnis = T_Test_Rückgabe(df,Zielvariable,Zielvariable_good,Zielvariable_bad,Zielvariable_Grenze)
    X_Ergebnis,X_Tabelle = chi_square_all_Rückgabe(df,Zielvariable,Zielvariable_Grenze) 
    
    # Tabelle
    
    Ergebnistabelle = T_Ergebnis + X_Ergebnis
    Ergebnistabelle = sorted(Ergebnistabelle,key=lambda x: x[1])
    pd.options.display.float_format = '{:,.2f}'.format
    Tabelle = pd.DataFrame(Ergebnistabelle, columns=['Variable', 'p-Wert',"X²;T-Wert",u"\u03bc_größer_grenze",u"\u03C3_größer_grenze",u"\u03bc_kleiner_grenze",u"\u03C3_kleiner_grenze","  dof  ","Testverfahren"])
    
    display(Tabelle)
    
    Cor_tab(df,max=10)
    
    # Ausgabe X²-Test
    
    if Zielvariable_bad=="":
        if Zielvariable_Grenze==0:
            Bad = "< "+ "{:.1f}".format(df[Zielvariable].values.mean())
            Good = "> "+ "{:.1f}".format(df[Zielvariable].values.mean())
        else:
            Bad = "< "+ f"{Zielvariable_Grenze}"
            Good = "> "+ f"{Zielvariable_Grenze}" 
    else:
        Bad = Zielvariable_bad
        Good = Zielvariable_good  
    
    if Anzahl_an_gewünschten_Variablen==None:
        Anzahl = Tabelle["Variable"].values
    else:
        Anzahl = Tabelle.iloc[0:Anzahl_an_gewünschten_Variablen]["Variable"].values
    
    
    for i in range(len(X_Tabelle)):
        if ((X_Tabelle[i][0]).index.name) in Anzahl:
            a = pd.DataFrame(X_Tabelle[i][0]) 
            b = pd.DataFrame(X_Tabelle[i][1])
            b=b.set_index(a.index)
            merge = pd.concat([a,b],axis=1)
            merge.columns=[Good,Bad,f"{Good}_erwartet",f"{Bad}_erwartet"]
            
            merge[f"{Good}%"] = merge[Good].combine(merge[f"{Good}_erwartet"], min) / merge[Good].combine(merge[f"{Good}_erwartet"], max)*100#merge["True"]/merge["True_erwartet"]*100
            merge[f"{Bad}%"] = merge[Bad].combine(merge[f"{Bad}_erwartet"], min) / merge[Bad].combine(merge[f"{Bad}_erwartet"], max)*100
            # merge = merge[["True","True_erwartet","False","False_erwartet","True%","False%"]]
            merge = merge.reindex(columns=merge.columns[[0,2,4,1,3,5]])
            merge = merge.rename_axis(Zielvariable,axis="columns")
            print(u"\u03c1 wert = {:.2f}".format(X_Ergebnis[i][1]),X_Ergebnis[i][1])
            print(u"\u03c7² = {:.2f}".format(X_Ergebnis[i][2]))
            print(f"dof = {X_Ergebnis[i][7]}")
            
            display(merge)
            print()
    
    for i in range(len(T_Ergebnis)):
        if ((T_Ergebnis[i][0])) in Anzahl:
            t=T_Ergebnis[i]
            figure,axis=plt.subplots()
            
            print(t[0],"-",Zielvariable,":","\nt =",'{:.1f}'.format(abs(t[2])),"\np =",'{:.1f}'.format(t[1]),t[1])
            print(Good,": ",u"\u03bc \u2248 ",'{:.4f}'.format(t[3]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(t[4]))
            print(Bad,": ",u"\u03bc \u2248 ",'{:.4f}'.format(t[5]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(t[6]))      
            axis.bar(x=np.arange(2),height=[t[3],t[5]],tick_label=[Good, Bad],color="gray",yerr=[[0,0],[t[4],t[6]]],ecolor="gray",width=0.4,capsize=4)
            axis.set_title(f"{t[0]} vs. {Zielvariable}")
            axis.set_ylabel(f"Durchschnitt {t[0]}");
            axis.set_xlabel(Zielvariable);
            plt.tight_layout()
            plt.show()
    
    ## Ausgabe
    if Anzahl_an_gewünschten_Variablen==None:
        return Tabelle,X_Tabelle,T_Ergebnis
    else:
        return Tabelle.iloc[0:Anzahl_an_gewünschten_Variablen],X_Tabelle,T_Ergebnis
def make_scatter_plot(x,y, width:int=8, height:int=6, xlabel:str="", ylabel:str="" ,title:str="", color="gray", filename:str="fig.png"):
    """Outputs a scatter plot
    
    Args:
        x: array with x-values
        y: array with y-values
        xticks (optional): list with values to use as x-ticks
        yticks (optional): list with values to use as y-ticks
        xlabel (optional): str with xlabel
        ylabel (optional): str with ylabel
        title (optional): str with title
        color (optional): str or tuple consisting of rgb values for color
        filename (optional): str with filename or path + filename
        
    Returns:
        Outputs the graph and saves it
        
    Raises:
        None
    """
    figure,axes=plt.subplots(figsize=(width,height), ncols=1)
           
    axes.set_xlabel(xlabel, );

    axes.set_ylabel(ylabel);
    axes.title.set_text(title)    

    #plt.xticks(rotation="vertical")
    axes.scatter(x=x, y=y, color=color,);

    plt.savefig(filename)    

class DataVisAnalyse:
    def __init__(self,df, classification_column, good, bad):
        """Creates Statistical Analysis (x2, pearson and student_t) 

        Args:
            df (pd.DataFrame): your DataFrame
            classification_column (str): column name as string
            good (str): good value
            bad (str): bad value    
        """
        
        
        self.df = df
        self.classification_column = classification_column
        self.good = good
        self.bad = bad
        self.dfStats = 0
    
    def __str__(self):
        print("""Konventionen:
              1. import datavis as dv
              2. analyser = dv.DataVisAnalyse(df:pd.DataFrame, classification_column:str, good:str, bad:str) 
              3. analyser.get_most_relevant(list_with_columns_to_ignore:list, features_to_inspect:int=5)
              4. analyser.get_corr_table()
              5. for each column analyser.get_chi2(column_name, df:None) # uses df provided in dv.DataVisAnalyse if df is None
              6. analyser.getT_Test(df:None)
              
              or 
              1. import datavis as dv
              2. analyser = dv.DataVisAnalyse(df:pd.DataFrame, classification_column:str, good:str, bad:str) 
              3. analyser.get_all()
              """)
    
    def get_most_relevant(self, ignore, n=5):
        """Get a df with most relevant variables
        
        Args:
            ignore (list): list with column_names which to ignore
            n (int): features to inspect
        """
        dfGood=self.df[self.df[self.classification_column]==self.good]
        dfBad=self.df[self.df[self.classification_column]==self.bad]
        statistics=[]
        for var in self.df.dtypes.keys():
            if var in ignore:
                continue
            dataType=self.df.dtypes[var]
            #print("var",var,"as",dataType)
            if dataType=="object":
                #X²-Test
                dfContingencyTable=pd.crosstab(self.df[var],self.df[self.classification_column])
                chi2,p,dof,exp=chi2_contingency(dfContingencyTable)
                #print("chi2",self.nice(chi2),"p",self.nice(p))
                statistics.append([var,"X²",chi2,p])
            else:
                #Student-Test
                t,p=ttest_ind(dfBad[var],dfGood[var])
                #print("t",self.nice(t),"p",self.nice(p))
                statistics.append([var,"t",t,p])
        dfStatistics=pd.DataFrame(statistics,columns=["Name","Test","stat","p"])    
        pd.options.display.float_format = '{:.2f}'.format
        dfStatistics.sort_values(by="p",inplace=True)
        display(HTML(dfStatistics[dfStatistics["p"] < 0.05].to_html())) 
        return dfStatistics[dfStatistics["p"] < 0.05]    
        
    def nice(aValue): return "{:.2f}".format(aValue)    
    
    def get_prcnt_diff(self, val1, val2):
        if val1>=val2:
            return f"{val2/val1*100:.2f}%"
        return f"{val1/val2*100:.2f}%"
    
    
    def get_corr_table(self, max:int=10, df=None):
        """[numerisch] - [numerisch]\n
        Stellt die stärksten Korrelationen zwischen intervallskalierten Variablen sortiert visuell dar.\n 
        Größeres p --> Größere Abhängigkeit
        @Variabel df: Zugehöriges DataFrame
        @Variabel max: Maximale Anzahl der Korrelationen
        """
        print("--------------------------------------------------")
        print("|               Korrelationstabelle              |")
        print("--------------------------------------------------")
        
        if df is None: df = self.df
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
        #print(correlations)    
        figure,axis=plt.subplots()
        plt.xticks(rotation=90)
        axis.bar(x=np.arange(len(heights)),height=heights,tick_label=labels,color="gray")
        axis.set_ylabel("Pearson Correlation p");
        plt.show()
    
    
    
    def get_chi2(self, varname, df=None):
        """
        [kategorisch] - [kategorisch]
        """
        if df is None: df = self.df
        contingencyTable=pd.crosstab(df[varname],df[self.classification_column])
        chi2,pval,dof,expectedFreq=chi2_contingency(contingencyTable)
        if pval<0.05:
            print(varname,"--",self.classification_column,":","chi2=",'{:.1f}'.format(chi2),"p=",'{:.5f}'.format(pval),"dof=",dof)#show pval in non-scienetific expression
            #print("expected values\n",expectedFreq)
            expectedFreq = np.array(expectedFreq)
            contingencyTable[f"exptected{self.good}"] = expectedFreq[:,0]
            contingencyTable[f"exptected{self.bad}"] = expectedFreq[:,1]
            diffs = [self.get_prcnt_diff(contingencyTable[self.good][i], contingencyTable[f"exptected{self.good}"][i]) for i in range(len(contingencyTable[self.good]))]
            contingencyTable[f"procentualDifference{self.good}"] = diffs
            diffs = [self.get_prcnt_diff(contingencyTable[self.bad][i], contingencyTable[f"exptected{self.bad}"][i]) for i in range(len(contingencyTable[self.bad]))]
            contingencyTable[f"procentualDifference{self.bad}"] = diffs
        else:
            print(f"Es gibt keinen stat. Zusammenhang p>0.05 bei {varname} --> p={pval};chi2={chi2}")
        display(HTML(contingencyTable.to_html()))    
        return contingencyTable
    
    
    def get_T_Test(self, df=None):
        """
        [metrisch] - [kategorisch]\n
        Stellt die unterschiedlichen stat. Abhängigkeiten mit Mittelwert und Standardabweichung visuell dar
        @self.classification_column: Nur zwei Ausprägungen
        @return [Variable,p-Wert,t-wert,mü-gut, sigma-gut, mü-schlecht, sigma-schlecht]
        """
        
        print("--------------------------------------------------")
        print("|                     T-Test                     |")
        print("--------------------------------------------------")
        
        P_Wert = []
        if df is None: df = self.df
        subset_good=df[df[self.classification_column]<=self.good]
        subset_bad=df[df[self.classification_column]>=self.bad]
        
        for testvar in df.keys():
            if df.dtypes[testvar]=="int64" or df.dtypes[testvar]=="float64":
                tval,pval=ttest_ind(subset_good[testvar],subset_bad[testvar])
                print(testvar,"-",self.classification_column,":","\nt =",'{:.1f}'.format(abs(tval)),"\np =",'{:.1f}'.format(pval),pval)
                
                
                
                figure,axis=plt.subplots()
                heights=[subset_good[testvar].mean(),subset_bad[testvar].mean()]
                stds=[[0,0],[subset_good[testvar].std(),subset_bad[testvar].std()]]
                
                P_Wert.append((testvar,pval,tval,heights[0],stds[1][0],heights[1],stds[1][1],"","T-Test"))
                
                print(self.good,": ",u"\u03bc \u2248 ",'{:.4f}'.format(heights[0]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(stds[1][0]))
                print(self.bad,": ",u"\u03bc \u2248 ",'{:.4f}'.format(heights[1]),",",u"\u03C3 \u2248 \u00B1",'{:.4f}'.format(stds[1][1]))      
                axis.bar(x=np.arange(len(heights)),height=heights,tick_label=[self.good, self.bad],color="gray",yerr=stds,ecolor="gray",width=0.4,capsize=4)
                axis.set_title(f"{testvar} vs. {self.classification_column}")
                axis.set_ylabel(f"Durchschnitt {testvar}");
                axis.set_xlabel(self.classification_column);
                plt.tight_layout()
                plt.show()
        return P_Wert
    
    
    def get_all(self):
        """Eine rundumanalyse eines Datensatzes in statistischer Form
        """
        print("                                   --------------------------------------------------")
        print("                                   |                     Analyse                    |")
        print("                                   --------------------------------------------------")
        res = self.get_most_relevant([self.classification_column])
        self.dfStats = res
        self.get_corr_table()
        for i in self.dfStats.Name:
            if self.df.dtypes[i] == "object":
                self.get_chi2(i)
        dfT_Test = self.df[list(self.dfStats.Name) + [self.classification_column]]        
        self.get_T_Test(df=dfT_Test)
        
        
        
            