from utils import *
from combined_metrics import *
from statistics import mean

fp_label_xml='/home/hungphd/git/LearnJAX/doe/data-all/label-split/test.json'
fp_predicted_xml='/home/hungphd/git/LearnJAX/doe/data-all/results/baselines/test.Qwen2.5-3B-Instruct.json'
# fp_predicted_xml='/home/hungphd/git/LearnJAX/doe/data-all/results/baselines/test.Llama-3.1-8B.json'

list_scores=[]
list_text_scores=[]
list_tag_scores=[]
json_labels=load_list_from_file(fp_label_xml)
json_predicts=load_list_from_file(fp_predicted_xml)

for i in range(0,len(json_labels)):
    score_lbl=0.0
    predict_item=json_predicts[i]
    label_item=json_labels[i]
    # print('{}\n\nlabel: {}\n\npredict: {}'.format(i,str(label_item['standard_xml']),str(predict_item['predicted_xml'])))
    score_obj=combined_similarity(str(label_item['standard_xml']),str(predict_item['predicted_xml']))
    list_scores.append(score_obj['combined_similarity'])
    list_text_scores.append(score_obj['text_similarity'])
    list_tag_scores.append(score_obj['tag_similarity'])

print('score avg comb {} text {} tag {}'.format(mean(list_scores),mean(list_text_scores),mean(list_tag_scores)))
