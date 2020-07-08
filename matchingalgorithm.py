import pandas as pd
from nltk import pos_tag, word_tokenize
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, lit
from pyspark.sql.types import LongType
import nltk
from fuzzywuzzy import fuzz

tempDF = spark.read.option("header", "true").csv(file)     
temp2DF = spark.read.option("header", "true").csv(file2)

temp2DF = temp2DF.dropna(subset = 'AUG_PRAC_COMPANY_NAME')
temp2DF.count()

OKY_list = tempDF.select('addr_ln_1', 'st_cd', 'bus_nm', 'ZIP5_CD', 'CITY_NM').collect()
LN_list = temp2DF.select('PRAC1_PRIMARY_ADDRESS_IN', 'PRAC1_STATE_IN', 'PRAC1_ZIP_IN', 'AUG_PRAC_COMPANY_NAME', 'PRAC1_CITY_IN').collect()
OKY_addr_busnm = [(row['addr_ln_1'], row['CITY_NM'], row['st_cd'], row['ZIP5_CD'], row['bus_nm']) for row in OKY_list]
LN_addr_busnm = [(row['PRAC1_PRIMARY_ADDRESS_IN'], row['PRAC1_CITY_IN'], row['PRAC1_STATE_IN'], row['PRAC1_ZIP_IN'], row['AUG_PRAC_COMPANY_NAME']) for row in LN_list]

real_bus_keywords = ['PHARMACY', 'OFFICE', 'CENTER', 'MEDICAL', 'HEALTH', 'CARE', 'HEALTHCARE', 'HOSPITAL', 'CLINIC', 'FAMILY', 'DRUG', 'ASSOCIATES', 'GROUP', 'SERVICES', 'SURGERY', 'HOME', 'INC', 'MEDICINE', 'COMMUNITY']

def actual_matching_process(name, name2):
  if len(name[0].split()[0]) == 2:
    if fuzz.ratio(name[0].split()[0], name2[0].split()[0])>=50:
      pass
    else:
      return 0
  elif len(name[0].split()[0]) == 3:
    if fuzz.ratio(name[0].split()[0], name2[0].split()[0])>=67:
      pass
    else:
      return 0
  elif len(name[0].split()[0]) == 4:
    if fuzz.ratio(name[0].split()[0], name2[0].split()[0])>=75:
      pass
    else:
      return 0
  elif len(name[0].split()[0]) == 5:
    if fuzz.ratio(name[0].split()[0], name2[0].split()[0])>=80:
      pass
    else:
      return 0
  else:
    pass
  bus_tokens = name[4].split()
  bus_tokens2 = name2[4].split()
  bus_keyword_match_list = []
  for i in bus_tokens:
    if i in bus_tokens2 and i in real_bus_keywords:
      bus_keyword_match_list.append(i)
  if len(bus_keyword_match_list)>0:
    return fuzz.ratio([x.replace(bus_keyword_match_list[0], '').strip() for x in name], [x.replace(bus_keyword_match_list[0], '').strip() for x in name2])
  else:
    return fuzz.ratio(name,name2)

# def actual_matching_process(name, name2, min_score=0):
#   max_addr_score = 0
#   max_addr = ''
#   max_busnm_score = 0
#   max_busnm = ''
#   if len(name[0].split())>1:
#     if name[0].split()[1] not in ['N','E','W','S','NE','NW','SE','SW','NORTH','EAST','WEST','SOUTH','NORTHEAST','NORTHWEST','SOUTHEAST','SOUTHWEST']:
#       statement = name[0][:4] == name2[0][:4]
#     else:
#       if len(name[0].split()[0]) == 1:
#         statement = name[0][:3] == name2[0][:3]
#       else:
#         statement = name[0][:4] == name2[0][:4]
#   else:
#     statement = name[0][:4] == name2[0][:4]
#   if statement:
#     addr_tokens = [i for i in name[0].split()]
#     addr_tokens2 = [i for i in name2[0].split()]
#     addr_keyword_match_list = []
#     for i in addr_tokens:
#       if i in addr_tokens2 and i in real_addr_keywords:
#         addr_keyword_match_list.append(i)
#     if len(addr_keyword_match_list)>0:
#       addr_score = fuzz.ratio(separate_numbers_and_words(name[0])[1].replace(addr_keyword_match_list[0], '').strip(), separate_numbers_and_words(name2[0])[1].replace(addr_keyword_match_list[0], '').strip())
#     else:
#       addr_score = fuzz.ratio(separate_numbers_and_words(name[0])[1].strip(), separate_numbers_and_words(name2[0])[1].strip())
#     if addr_score >= min_score:
#       max_addr = name2[0]
#       max_addr_score = addr_score
#       bus_tokens = [i for i in name[4].split()]
#       bus_tokens2 = [i for i in name2[4].split()]
#       bus_keyword_match_list = []
#       for i in bus_tokens:
#         if i in bus_tokens2 and i in real_bus_keywords:
#           bus_keyword_match_list.append(i)
#       if len(bus_keyword_match_list)>0:
#         bus_score = fuzz.ratio(name[4].replace(bus_keyword_match_list[0], '').strip(), name2[4].replace(bus_keyword_match_list[0], '').strip())
#       else:
#         bus_score = fuzz.ratio(name[4].strip(), name2[4].strip())
#       if bus_score >= (min_score+5):
#         max_busnm = name2[4]
#         max_busnm_score = bus_score
        
#   return (max_addr, max_addr_score, max_busnm, max_busnm_score)

def match_name(name, list_names, min_score=0):
    score = 0
    best_score = 0
    best_match = ''
    for name2 in list_names:
      if name[2].strip()=='' or name2[2].strip()=='':
        score = actual_matching_process(name, name2)
      else:
        if name[2].strip()==name2[2].strip():
          if name[1].strip()=='' or name2[1].strip()=='':
            score = actual_matching_process(name, name2)
          else:
            if fuzz.ratio(name[1].strip(), name2[1].strip())>=80:
              score = actual_matching_process(name, name2)
      if score>=min_score:
        best_score = score
        best_match = name2
        break
    return (best_score, best_match)

dict_list = []
for name in OKY_addr_busnm:
  match = match_name(name, LN_addr_busnm, 75)
  dict_ = {}
  dict_.update({"OKY_entry" : name})
  dict_.update({"LN_entry" : match[1]})
  dict_.update({"Match_score" : match[0]})
  if match[1]!='' or match[0]!=0:
    dict_list.append(dict_)
merge_table = pd.DataFrame(dict_list)
display(merge_table)

# dict_list = []
# for name in OKY_addr_busnm:
#   match = match_name(name, LN_addr_busnm, 45)
#   dict_ = {}
#   dict_.update({"OKY_street" : name[0]})
#   dict_.update({"LN_street" : match[0]})
#   dict_.update({"Addr_match_score" : match[1]})
#   dict_.update({"OKY_busnm" : name[4]})
#   dict_.update({"LN_busnm" : match[2]})
#   dict_.update({"Busnm_match_score" : match[3]})
#   if match[2]!='' or match[3]!=0:
#     dict_list.append(dict_)
# merge_table = pd.DataFrame(dict_list)
# display(merge_table)

sparkdf = spark.createDataFrame(merge_table)
sparkdf.coalesce(1).write.option("header", "true").save("dbfs:/FileStore/df/test.csv")
display(merge_table)
dbutils.fs.rm("dbfs:/FileStore/df",True)
