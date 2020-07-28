from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb 
import pandas as pd
from utils.sql_utils_py3 import _get_sql75_connection
from scipy import stats
import os.path
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from scipy.stats.mstats import winsorize
from PIL import Image
import sys
from PyPDF2 import PdfFileMerger
import pipeline_config
from pipeline_config import PipelineConfig, load_config_file
import seaborn as sns
from scipy.stats import gaussian_kde

filename = 'prod_sales_pipeline.yml'
PipelineConfig(filename)
FIGSIZE = [20,20]
ALPHA = 0.5

class featureToPlot():
    """ Class containing feature to plot plus attributes

        Attributes:
          feature_name [str]: Name of the feature
          param_list [list]: values of the positive bin counts, negative bin counts and historgram edges
    """
    def __init__(self, feature_name=None, target_name=None, categorical_feature_flg=None,
                impute_strategy=None, fill_nan_value=None, log_transform_flg=None,winsorize_flg=None,
                winsorize_limits=None,scaler_type=None,flag_nan_values_flg=None,drop_nan_flg=None):
        self.feature_name = feature_name
        self.target_name = target_name
        self.column_headers = None
        self.data = None #filled by method load_feature_data
        self.param_list = [] #values of the positive bin counts, negative bin counts and historgram edges
        self.winsorize_limits = None #in future will read from PipelineConfig()
        self.categorical_feature_flg = categorical_feature_flg
        self.impute_strategy = impute_strategy
        self.fill_nan_value = fill_nan_value
        self.log_transform_flg = log_transform_flg
        self.winsorize_flg = winsorize_flg
        self.winsorize_limits = winsorize_limits
        self.scaler_type = scaler_type
        self.flag_nan_values_flg = flag_nan_values_flg
        self.drop_nan_flg = drop_nan_flg
        
    def load_feature_data(self, use_local_file=False):
        """ Fetches data from SQL, can create cached file if user plans to get data more than once 
            
            Arguments:

            Returns:
                  updates self.data a pandas dataframe with 2 columns: feature value and target avlue
        """
        filename = f'{self.feature_name}_cached_data.csv'
        if not use_local_file or not os.path.exists(filename):
            conn = _get_sql75_connection()
            sql_str = f'SELECT {self.feature_name}, {self.target_name} FROM {PipelineConfig().table_name}'\
                f' WHERE {self.feature_name} IS NOT NULL AND {self.target_name} IS NOT NULL'
            data = pd.read_sql(sql_str, conn)
            conn.dispose()
            #Positive/negative class identified in config_file
            data.loc[:,'TARGET_FLG'] = data.loc[:,self.target_name] > 0
            data.to_csv(filename, index=False)
        else:
            print('Used Data')
            data = pd.read_csv(filename)
        if data.empty:
            raise ValueError('Data is empty')
        self.data = data

    def feature_transform(self):
        """ Applies log, winsorize, and scaling transformations to data where indicated in yml file """
        if self.log_transform_flg:
            self.data['LOG10_'+self.feature_name] = np.log10(self.data[self.feature_name] - self.data[self.feature_name].min()+1)
        else:
            self.data['LOG10_'+self.feature_name] = self.data[self.feature_name]  #not actually log transforming
        #use winsorize_limits from config_file
        if self.winsorize_flg:
            self.data['WINS_'+self.feature_name] = winsorize(self.data['LOG10_'+self.feature_name],limits = self.winsorize_limits)
        else:
            self.data['WINS_'+self.feature_name] = self.data['LOG10_'+self.feature_name]
        if self.scaler_type:
            if self.scaler_type == 'dynamic':
                # get feature raw median value (before log/winsorize)
                ftr_median = self.data[self.feature_name].median()
                # use robust scaler if value is near zero, else std scaler
                not_quite_zero = 1e-5
                if abs(ftr_median) <= not_quite_zero:
                    self.data['SCLD_'+self.feature_name] = self.data['WINS_'+self.feature_name] - self.data['WINS_'+self.feature_name].median()\
                        /(self.data['WINS_'+self.feature_name].max() - self.data['WINS_'+self.feature_name].min())
                else:
                    scaler = StandardScaler()
                    self.data['SCLD_'+self.feature_name] = scaler.fit_transform(self.data['WINS_'+self.feature_name].values.reshape(-1,1))
            elif self.scaler_type == 'MedianMinMaxScaler':
                self.data['SCLD_'+self.feature_name] = self.data['WINS_'+self.feature_name] - self.data['WINS_'+self.feature_name].median()\
                /(self.data['WINS_'+self.feature_name].max() - self.data['WINS_'+self.feature_name].min())
            else:
                if self.scaler_type in 'StandardScaler':
                    scaler = StandardScaler()
                elif self.scaler_type in 'MinMaxScaler':
                    scaler = MinMaxScaler()
                elif self.scaler_type in 'MaxAbsScaler':
                    scaler = MaxAbsScaler()
                elif self.scaler_type in 'RobustScaler':
                    scaler = RobustScaler()
                elif self.scaler_type in 'PolynomialFeatures':
                    scaler = PolynomialFeatures()
                elif self.scaler_type in 'Normalizer':
                    scaler = Normalizer()
                elif self.scaler_type in 'Binarizer':
                    scaler = Binarizer()
                elif self.scaler_type in 'KernelCenterer':
                    scaler = KernelCenterer()
                elif self.scaler_type in 'MaxAbsScaler':
                    scaler = MaxAbsScaler()
                elif self.scaler_type in 'QuantileTransformer':
                    scaler = QuantileTransformer()
                else:
                    raise KeyError(f'Scaler type {self.scaler_type} not supported')
                self.data['SCLD_'+self.feature_name] = scaler.fit_transform(self.data['WINS_'+self.feature_name].values.reshape(-1,1))
        else:
            self.data['SCLD_'+self.feature_name] = self.data['WINS_'+self.feature_name]
                

    #Creates histogram of positive and negative classes of target variable     
    def create_histogram(self, variable_name, target_name, nrows, idx):
        ftr_min = self.data[variable_name].min()
        ftr_max = self.data[variable_name].max()
        ftr_avg = self.data[variable_name].mean()
        ftr_stdev = self.data[variable_name].std()
        ftr_median = self.data[variable_name].median()
        num_bins = 15
        bin_strt = ftr_min
        bin_end = ftr_max
        bin_size = (bin_end - bin_strt)/num_bins
        #class definition from config file
        positive_clss_definition = f'{target_name} > 0'
        negative_clss_definition = f'{target_name} <= 0'
        #Create axis for suplot 1
        ax1 = plt.subplot(nrows,4,idx)
        plt.title(f'{self.column_headers[idx-1]}\nHistogram of {target_name} vs \n{variable_name}')
        plt.xlabel(f'{variable_name}')
        plt.ylabel('Count')
        # Create successful campaign histogram for target variable
        pos_vals = np.array(self.data[variable_name][self.data['TARGET_FLG']==1].values)
        pos_vals = pos_vals[np.isfinite(pos_vals)]
        n_positive, bins_positive, patches_positive = plt.hist(pos_vals, bins=np.arange(bin_strt, 
        bin_end+bin_size, bin_size), facecolor='b', alpha=ALPHA, label=positive_clss_definition)
        # Create NON-successful histogram
        neg_vals = np.array(self.data[variable_name][self.data['TARGET_FLG']==0].values)
        neg_vals = neg_vals[np.isfinite(neg_vals)]
        n_negative, bins_negative, patches_negative = plt.hist(neg_vals, bins=bins_positive, 
        facecolor='r',alpha=ALPHA, label=negative_clss_definition)
        # Set plot x limits
        plt.xlim([bins_positive.min(), bins_positive.max()])
        # In the case when the maximum bin contains greater than 10 times the minimum bin
        # use a log y scale to better visualize the distribution
        if max(n_positive)/min(n_positive)> 10:
         plt.yscale("log")
        # Add a legend
        plt.legend()
        self.param_list.extend([n_positive, n_negative, bins_positive])
        if 'N/A' not in self.column_headers[idx-1]:
            plt.figtext(0.15 + 0.25 * (idx-1), 0.9, 'Mean: {0:.3f} \nStd Dev: {1:.3f} \nMedian: {2:.3f} \nMax: {3:.3f} \n'\
            'Min: {4:.3f}'.format(ftr_avg, ftr_stdev, ftr_median, ftr_max, ftr_min),fontsize=10)
        else:
            ax1.cla()
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            plt.title(f'{self.column_headers[idx-1]}')

               #Creates probability plot of positive class target variable vs feature
    def create_prob_plot(self, variable_name, target_name, nrows, idx):     
    # Create probability plot
        ax2 = plt.subplot(nrows,4,idx)
        n_total = self.param_list[0] + self.param_list[1]
        proba_positive = self.param_list[0] / n_total
        sigma_positive = np.sqrt(self.param_list[0])
        sigma_total = np.sqrt(n_total)
        error_proba = proba_positive * np.sqrt((sigma_positive/self.param_list[0])**2 + (sigma_total/n_total)**2)
        bin_center = (self.param_list[2][:-1]+self.param_list[2][1:])/2
        plt.title(f'Probability of Positive {target_name} vs \n{variable_name}')
        plt.xlabel(f'{variable_name}')
        plt.ylabel('Probability')
        plt.errorbar(bin_center, proba_positive, yerr=error_proba,fmt='.')
        plt.xlim([self.param_list[2].min(), self.param_list[2].max()])
        self.param_list = self.param_list[3:]
        if 'N/A' in self.column_headers[idx-5]:
            ax2.cla()
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

    def create_corr_plot(self, variable_name, target_name, nrows, idx):
                   #Create correlation plot
        ax3 = plt.subplot(nrows,4,idx)
        plt.title(f'Correlation Between {target_name} and \n{variable_name}')
        plt.xlabel(f'{variable_name}')
        plt.ylabel(f'{target_name}')
        pos_x_clss_df = self.data[variable_name][self.data['TARGET_FLG']==1]
        neg_x_clss_df = self.data[variable_name][self.data['TARGET_FLG']==0]
        pos_y_clss_df = self.data[target_name][self.data['TARGET_FLG']==1]
        neg_y_clss_df = self.data[target_name][self.data['TARGET_FLG']==0]
        pos_pearson_r, pos_corr_p_val =  stats.pearsonr(pos_x_clss_df,pos_y_clss_df)
        neg_pearson_r, neg_corr_p_val =  stats.pearsonr(neg_x_clss_df,neg_y_clss_df)
        #The correlation p-value indicates the probability of an uncorrelated system having a Pearson 
        #correlation at least as extreme as the calculated r
        plt.plot(pos_x_clss_df,pos_y_clss_df, 'g.',label='{}>0'.format(target_name))
        plt.plot(neg_x_clss_df,neg_y_clss_df, 'b.',label='{}<=0'.format(target_name))
        #Fitting linear regression to positive and negative classes
        z1 = np.polyfit(pos_x_clss_df,pos_y_clss_df, 1)
        p1 = np.poly1d(z1)
        plt.plot(pos_x_clss_df, p1(pos_x_clss_df),'r-', label='y={0:.3f}x+{1:.3f}(pos)'.format(z1[0],z1[1]))
        z2 = np.polyfit(neg_x_clss_df,neg_y_clss_df, 1)
        p2 = np.poly1d(z2)
        plt.plot(neg_x_clss_df, p2(neg_x_clss_df),'c-', label='y={0:.3f}x+{1:.3f}(neg)'.format(z2[0],z2[1]))
        plt.legend(loc='upper right')
        plt.figtext(0.15 + 0.25 * (idx-13), 0.15, 
                                    'Pearsons r pos = {0:.5f}\nR-squared pos = {1:.5f} '\
                                   '\npos_p_val = {2:.5f}'.format(pos_pearson_r, pos_pearson_r**2,pos_corr_p_val), fontsize=10)
        plt.figtext(0.15 + 0.25 * (idx-13), 0.05,  
                                    'Pearsons r neg = {0:.5f} \nR-squared neg = {1:.5f} '\
                                   '\nneg_p_val = {2:.5f}'.format(neg_pearson_r, neg_pearson_r**2,neg_corr_p_val), fontsize=10)

               #Creates boxplot of positive and negative classes of target variable          
    def create_boxplot(self, variable_name, target_name, nrows, idx):
        ax3 = plt.subplot(nrows,4,idx)
        # xy = np.vstack([self.data['TARGET_FLG'],self.data[variable_name]])
        # z = gaussian_kde(xy)(xy)
        # pdb.set_trace()
        sns.boxplot(x = 'TARGET_FLG', y = variable_name, data = self.data)
        sns.stripplot(x = 'TARGET_FLG', y = variable_name, data = self.data, alpha = 0.3, color = 'black', jitter = False)
        ax3.set_xticklabels(['Negative','Positive'])
        plt.ylabel(variable_name)
        plt.title(f'Boxplots of Pos/Neg {target_name} vs \n{variable_name}')
        if 'N/A' in self.column_headers[idx-13]:
            ax3.cla()
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            
               #Creates cdf for positive and negative classes of target variable
    def create_cdf(self, variable_name, target_name, nrows, idx):
        num_bins = 15
        #create attributes for below values
        bin_strt = self.data[variable_name].min()
        bin_end = self.data[variable_name].max()
        bin_size = (bin_end - bin_strt)/num_bins
        pos_vals = np.array(self.data[variable_name][self.data['TARGET_FLG']==1].values)
        pos_vals = pos_vals[np.isfinite(pos_vals)]
        neg_vals = np.array(self.data[variable_name][self.data['TARGET_FLG']==0].values)
        neg_vals = neg_vals[np.isfinite(neg_vals)]
        ax4 = plt.subplot(nrows,4,idx)
        plt.title('CDF')
        plt.xlabel(f'{variable_name}')
        pos_hist, bin_edges = np.histogram(pos_vals, bins=num_bins+1, range =(bin_strt-bin_size,bin_end),
        density=True)
        cdf_pos = np.cumsum(pos_hist*np.diff(bin_edges))
        neg_hist, bin_edges = np.histogram(neg_vals, bins=num_bins+1, range =(bin_strt-bin_size,bin_end),
        density=True)
        cdf_neg = np.cumsum(neg_hist*np.diff(bin_edges))
        # x_list = np.arange(bin_strt-bin_size, bin_end, bin_size)
        # if len(x_list)>len(cdf_pos):
                     # x_list = x_list[1:]
        plt.plot(np.linspace(bin_strt-bin_size, bin_end, num_bins+1), cdf_pos, color='b', label=f'{target_name}>0')
        plt.plot(np.linspace(bin_strt-bin_size, bin_end, num_bins+1), cdf_neg, color='r', label=f'{target_name}<=0')
        plt.ylim([0,1])
        plt.xlim([bin_strt-bin_size, bin_end])
        ks_stat, ks_p_val = stats.ks_2samp(self.data[variable_name][self.data['TARGET_FLG']==1], self.data[variable_name]
        [self.data['TARGET_FLG']==0])
        if 'N/A' not in self.column_headers[idx-9]:
            plt.text(0.7, 0.8, 't-stat: %3.2f'%(ks_stat), transform=ax4.transAxes)
            plt.legend()
        else:
            ax4.cla()
            ax4.get_xaxis().set_visible(False)
            ax4.get_yaxis().set_visible(False)

               #Performs t test, welch test, KS test        
    def hypothesis_testing(self, feature_name):
        pos_variance = self.data[feature_name][self.data['TARGET_FLG']==1].std()**2
        neg_variance = self.data[feature_name][self.data['TARGET_FLG']==0].std()**2
        #Assuming equal variances
        equal_avg_p_val = stats.ttest_ind(self.data[feature_name][self.data['TARGET_FLG']==1], self.data[feature_name]
        [self.data['TARGET_FLG']==0])[1]
        #Assuming unequal variances
        welch_stat, welch_p_val = stats.ttest_ind(self.data[feature_name][self.data['TARGET_FLG']==1], self.data[feature_name]
        [self.data['TARGET_FLG']==0], equal_var=False)
        #If p value is <0.05, reject null hypothesis of equal means, else cannot reject null
        #Kolmogorov-Smirnow Test
        ks_stat, ks_p_val = stats.ks_2samp(self.data[feature_name][self.data['TARGET_FLG']==1], self.data[feature_name]
        [self.data['TARGET_FLG']==0])
        #Tests whether two samples are drawn from same distribution; if p value is <0.05, reject null hypothesis
        #of same distributions, else cannot reject null
        #Lillie Test
        return (equal_avg_p_val, welch_stat, welch_p_val, ks_stat, ks_p_val)

#Generates png of plots for each feature, then converts to pdf and merges them
def main():
    target_name = PipelineConfig().y_col
    pdfs = []
    #create column headers for feature plots using config file
    f_config = load_config_file(filename, 'feature_config')
    feature_dict = dict(zip(PipelineConfig().features, f_config['features']))
    
    for feature in feature_dict:
        # Collect feature attributes from config file
        categorical_feature_flg = feature_dict[feature]['categorical_feature_flg']
        impute_strategy = feature_dict[feature]['impute_strategy']
        fill_nan_value = feature_dict[feature]['fill_nan_value']
        log_transform_flg = feature_dict[feature]['log_transform_flg']
        winsorize_flg = feature_dict[feature]['winsorize_flg']
        winsorize_limits = feature_dict[feature]['winsorize_limits']
        scaler_type = feature_dict[feature]['scaler_type']
        flag_nan_values_flg = feature_dict[feature]['flag_nan_values_flg']
        drop_nan_flg = feature_dict[feature]['drop_nan_flg']
        
        #Initiate featureToPlot class
        ftrClss = featureToPlot(feature_name=feature, target_name=target_name, categorical_feature_flg=categorical_feature_flg,
                                impute_strategy=impute_strategy, fill_nan_value=fill_nan_value, log_transform_flg =log_transform_flg,winsorize_flg =winsorize_flg,
                                winsorize_limits =winsorize_limits,scaler_type=scaler_type,flag_nan_values_flg=flag_nan_values_flg,drop_nan_flg=drop_nan_flg)
                               
        #Load feature and target columns to a dataframe in ftrClss.data
        ftrClss.load_feature_data()
        
        column_headers = ['RAW FEATURE', 'LOG TRANSFORMED', 'WINSORIZED', 'SCALED']
        if not log_transform_flg:
            column_headers[1] = 'LOG TRANSFORM N/A'
        if not winsorize_flg:
            column_headers[2] = 'WINSORIZE N/A'
        if not scaler_type:
            column_headers[3] = 'SCALING N/A'
        else:
            column_headers[3] = f'{scaler_type.upper()} SCALED'
        print(f'Processing feature: {feature}')
        
        # Add column headers to ftrClss attributes
        ftrClss.column_headers = column_headers

        # Transform feature, adds tranforms columsn
        ftrClss.feature_transform()

        #Andy's logic for choosing scaler
        plt.figure(figsize=FIGSIZE)
        transformations = ['', 'LOG10_', 'WINS_', 'SCLD_']
        pdf_filename = f'campaign_results_vs_{feature}.pdf'
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        for idx,method in enumerate(transformations):
            ftrClss.create_histogram(method+feature,target_name,4,idx+1)
            ftrClss.create_prob_plot(method+feature,target_name,4,idx+5)
            ftrClss.create_cdf(method+feature,target_name,4,idx+9)
            ftrClss.create_boxplot(method+feature,target_name,4,idx+13)
        plt.tight_layout()
        plt.savefig(f'campaign_results_vs_{feature}.png')
        image1 = Image.open(f'campaign_results_vs_{feature}.png')
        im1 = image1.convert('RGB')
        im1.save(pdf_filename)
        pdfs.append(pdf_filename)
        plt.close()
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    #have saved filename in config file
    merger.write('store_churn_sales.pdf')
    merger.close()
    # print(data.head())
    
if __name__ == '__main__':
    main()
