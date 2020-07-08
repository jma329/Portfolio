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
from pipeline_config import PipelineConfig

pipeline_config.CONFIG_FILE_NAME = 'campaign_config.yml'
PipelineConfig()
FIGSIZE = [20,20]
ALPHA = 0.5
parameter_list = []

def load_feature_data(feature_name,target_name,use_local_file=False):
	filename = 'cached_data.csv'
	if not use_local_file or not os.path.exists(filename):
		conn = _get_sql75_connection()
		sql_str = f'SELECT {feature_name}, {target_name} FROM {PipelineConfig.table_name}'\
				  f' WHERE {feature_name} IS NOT NULL AND {target_name} IS NOT NULL'
		data = pd.read_sql(sql_str, conn)
		conn.dispose()
		data.to_csv(filename, index=False)
	else:
		print('Used Data')
		data = pd.read_csv(filename)
	if data.empty:
		raise ValueError('Data is empty')
	return data
		
def feature_transform(feature_name, data):
	data['LOG10_'+feature_name] = np.log10(data[feature_name] - data[feature_name].min()+1)
	winsorize_limits = .01
	data['WINS_'+feature_name] = winsorize(data['LOG10_'+feature_name],limits = winsorize_limits)
	if PipelineConfig.scaler_mapping[feature_name] == 'dynamic':
		# get feature raw median value (before log/winsorize)
		ftr_median = data[feature_name].median()
        # use robust scaler if value is near zero, else std scaler
		not_quite_zero = 1e-5
		if abs(ftr_median) <= not_quite_zero:
			data['MMM_'+feature_name] = data['WINS_'+feature_name] - data['WINS_'+feature_name].median()\
			/(data['WINS_'+feature_name].max() - data['WINS_'+feature_name].min())
		else:
			scaler = StandardScaler()
			data['STD_'+feature_name] = scaler.fit_transform(data['WINS_'+feature_name].values.reshape(-1,1))
	elif PipelineConfig.scaler_mapping[feature_name] in ('standard',None):
		scaler = StandardScaler()
		data['STD_'+feature_name] = scaler.fit_transform(data['WINS_'+feature_name].values.reshape(-1,1))
	elif PipelineConfig.scaler_mapping[feature_name] == 'MedianMinMaxScaler':
		data['MMM_'+feature_name] = data['WINS_'+feature_name] - data['WINS_'+feature_name].median()\
		/(data['WINS_'+feature_name].max() - data['WINS_'+feature_name].min())
	return data
	
def create_histogram(data, variable_name, target_name, nrows, idx):
	ftr_min = data[variable_name].min()
	ftr_max = data[variable_name].max()
	ftr_avg = data[variable_name].mean()
	ftr_stdev = data[variable_name].std()
	ftr_median = data[variable_name].median()
	num_bins = 15
	bin_strt = ftr_min
	bin_end = ftr_max
	bin_size = (bin_end - bin_strt)/num_bins
	positive_clss_definition = f'{target_name} > 0'
	negative_clss_definition = f'{target_name} <= 0'
	#Create axis for suplot 1
	ax1 = plt.subplot(nrows,4,idx)
	plt.title(f'Histogram of Campaign results vs \n{variable_name}')
	plt.xlabel(f'{variable_name}')
	plt.ylabel('Count')
	# Create successful campaign histogram for target variable
	pos_vals = np.array(data[variable_name][data['TARGET_FLG']==1].values)
	pos_vals = pos_vals[np.isfinite(pos_vals)]
	n_positive, bins_positive, patches_positive = plt.hist(pos_vals, bins=np.arange(bin_strt, 
	bin_end+bin_size, bin_size), facecolor='b', alpha=ALPHA, label=positive_clss_definition)
	# Create NON-successful histogram
	neg_vals = np.array(data[variable_name][data['TARGET_FLG']==0].values)
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
	plt.figtext(0.15 + 0.25 * (idx-1), 0.9, 'Mean: {0:.3f} \nStd Dev: {1:.3f} \nMedian: {2:.3f} \nMax: {3:.3f} \n'\
	'Min: {4:.3f}'.format(ftr_avg, ftr_stdev, ftr_median, ftr_max, ftr_min),fontsize=10)
	global parameter_list
	parameter_list.extend([n_positive, n_negative, bins_positive])

def create_prob_plot(data, variable_name, target_name, nrows, idx):	
# Create probability plot
	ax2 = plt.subplot(nrows,4,idx)
	global parameter_list
	n_total = parameter_list[0] + parameter_list[1]
	proba_positive = parameter_list[0] / n_total
	sigma_positive = np.sqrt(parameter_list[0])
	sigma_total = np.sqrt(n_total)
	error_proba = proba_positive * np.sqrt((sigma_positive/parameter_list[0])**2 + (sigma_total/n_total)**2)
	bin_center = (parameter_list[2][:-1]+parameter_list[2][1:])/2
	plt.title(f'Probability of Positive {target_name} vs \n{variable_name}')
	plt.xlabel(f'{variable_name}')
	plt.ylabel('Probability')
	plt.errorbar(bin_center, proba_positive, yerr=error_proba,fmt='.')
	plt.xlim([parameter_list[2].min(), parameter_list[2].max()])
	parameter_list = parameter_list[3:]

def create_corr_plot(data, variable_name, target_name, nrows, idx):
	#Create correlation plot
	ax3 = plt.subplot(nrows,4,idx)
	plt.title(f'Correlation Between {target_name} and \n{variable_name}')
	plt.xlabel(f'{variable_name}')
	plt.ylabel(f'{target_name}')
	pos_x_clss_df = data[variable_name][data['TARGET_FLG']==1]
	neg_x_clss_df = data[variable_name][data['TARGET_FLG']==0]
	pos_y_clss_df = data[target_name][data['TARGET_FLG']==1]
	neg_y_clss_df = data[target_name][data['TARGET_FLG']==0]
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
		
def create_cdf(data, variable_name, target_name, nrows, idx):
	num_bins = 15
	bin_strt = data[variable_name].min()
	bin_end = data[variable_name].max()
	bin_size = (bin_end - bin_strt)/num_bins
	pos_vals = np.array(data[variable_name][data['TARGET_FLG']==1].values)
	pos_vals = pos_vals[np.isfinite(pos_vals)]
	neg_vals = np.array(data[variable_name][data['TARGET_FLG']==0].values)
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
	x_list = np.arange(bin_strt-bin_size, bin_end, bin_size)
	if len(x_list)>len(cdf_pos):
		x_list = x_list[1:]
	plt.plot(x_list, cdf_pos, color='b', label=f'{target_name}>0')
	plt.plot(x_list, cdf_neg, color='r', label=f'{target_name}<=0')
	plt.ylim([0,1])
	plt.xlim([bin_strt-bin_size, bin_end])
	ks_stat, ks_p_val = stats.ks_2samp(data[variable_name][data['TARGET_FLG']==1], data[variable_name]
	[data['TARGET_FLG']==0])
	plt.text(0.7, 0.8, 't-stat: %3.2f'%(ks_stat), transform=ax4.transAxes)
	plt.legend()
	
def hypothesis_testing(data, feature_name):
	pos_variance = data[feature_name][data['TARGET_FLG']==1].std()**2
	neg_variance = data[feature_name][data['TARGET_FLG']==0].std()**2
	#Assuming equal variances
	equal_avg_p_val = stats.ttest_ind(data[feature_name][data['TARGET_FLG']==1], data[feature_name]
	[data['TARGET_FLG']==0])[1]
	#Assuming unequal variances
	welch_stat, welch_p_val = stats.ttest_ind(data[feature_name][data['TARGET_FLG']==1], data[feature_name]
	[data['TARGET_FLG']==0], equal_var=False)
	#If p value is <0.05, reject null hypothesis of equal means, else cannot reject null
	#Kolmogorov-Smirnow Test
	ks_stat, ks_p_val = stats.ks_2samp(data[feature_name][data['TARGET_FLG']==1], data[feature_name]
	[data['TARGET_FLG']==0])
	#Tests whether two samples are drawn from same distribution; if p value is <0.05, reject null hypothesis
	#of same distributions, else cannot reject null
	return (equal_avg_p_val, welch_stat, welch_p_val, ks_stat, ks_p_val)

def main():
	# features = ['ftr_num_items_in_campaign',
 # 'ftr_item_days_since_last_campaign',
 # 'ftr_cust_days_since_last_campaign',
 # 'ftr_cust_ftr_item_days_since_last_campaign',
 # 'ftr_baseline_compliance',
 # 'ftr_campaign_discount_dollars',
 # 'ftr_target_net_price',
 # 'ftr_target_deal_pct',
 # 'ftr_rebate_rate',
 # 'ftr_campaign_discount_percent',
 # 'ftr_dm_rnkng_cd',
 # 'ftr_wac',
 # 'bsln_sls_qty',
 # 'ftr_avg_order_qty',
 # 'ftr_time_since_last_order',
 # 'ftr_num_successful_campaigns_last_yr',
 # 'prcnt_num_successful_campaigns_last_yr']
	target_name = 'INC_SLS_QTY'
	pdfs = []
	for feature in PipelineConfig.features[:1]:
		print(f'Processing feature: {feature}')
		data = load_feature_data(feature,target_name)
		data = data.loc[:,[feature,target_name]]
		data = feature_transform(feature, data)
		data.loc[:,'TARGET_FLG'] = data.loc[:,target_name] > 0
		plt.figure(figsize=FIGSIZE)
		if PipelineConfig.scaler_mapping[feature] == 'dynamic':
			ftr_median = data[feature].median()
			not_quite_zero = 1e-5
			if abs(ftr_median) <= not_quite_zero:
				scaler_prefix = 'MMM_'
			else:
				scaler_prefix = 'STD_'
		elif PipelineConfig.scaler_mapping[feature] in ('standard',None):
			scaler_prefix = 'STD_'
		elif PipelineConfig.scaler_mapping[feature] == 'MedianMinMaxScaler':
			scaler_prefix = 'MMM_'
		transformations = ['', 'LOG10_', 'WINS_', scaler_prefix]
		pdf_filename = f'campaign_results_vs_{feature}.pdf'
		if os.path.exists(pdf_filename):
			os.remove(pdf_filename)
		for idx,method in enumerate(transformations):
			create_histogram(data,method+feature,target_name,4,idx+1)
			create_prob_plot(data,method+feature,target_name,4,idx+5)
			create_cdf(data,method+feature,target_name,4,idx+9)
			create_corr_plot(data,method+feature,target_name,4,idx+13)
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
	merger.write('campaign_features.pdf')
	merger.close()
	print(data.head())
    
if __name__ == '__main__':
    main()