from src.xml.combined_metrics import *

def read_text_file(file_path):
    """Reads and returns the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

fop_data='../../preprocess-dataset/sample-1/'
fp_label=fop_data+'answer.txt'
fp_org_mistral=fop_data+'org-mistral.xml'
fp_opt_mistral=fop_data+'answer.xml'

result_org = combined_similarity(read_text_file(fp_org_mistral), read_text_file(fp_label), alpha=0.7)
result_opt = combined_similarity(read_text_file(fp_opt_mistral), read_text_file(fp_label), alpha=0.7)

print('org acc {}'.format(result_org))
print('opt acc {}'.format(result_opt))