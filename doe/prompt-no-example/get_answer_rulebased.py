import copy

from doe.utils import *
from doe.combined_metrics import *
from statistics import mean
from rulebased import *

fp_label_xml='/home/hungphd/git/LearnJAX/doe/data-all/label-split/test.json'
fp_ouput_xml='/home/hungphd/git/LearnJAX/doe/data-all/results/rulebased/test.rulebased.json'

ensure_parent_dir(fp_ouput_xml)

list_outputs=[]
json_labels=load_list_from_file(fp_label_xml)


for i in range(0,len(json_labels)):
    score_lbl=0.0
    # predict_item=json_predicts[i]
    label_item=json_labels[i]
    if 'standard_xml' in label_item.keys():
        label_item.pop('standard_xml')
    # print('{}\n\nlabel: {}\n\npredict: {}'.format(i,str(label_item['standard_xml']),str(predict_item['predicted_xml'])))

    rule_obj=json_to_puboutages_xml(label_item)
    copied_item=copy.copy(label_item)
    copied_item['predicted_xml']=rule_obj
    list_outputs.append(copied_item)

save_list_to_file(list_outputs,fp_ouput_xml)