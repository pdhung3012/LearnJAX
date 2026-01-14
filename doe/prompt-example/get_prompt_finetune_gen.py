from utils import *
from doe.prompt_template import *
import csv

fp_examples='data-all/standard_xml_small.txt'
f1=open(fp_examples,'r')
str_example=f1.read()
f1.close()


def build_data_for_fine_tuning(item: Dict[str, Any]) -> Dict[str, str]:
    # print(item)
    str_answer=''
    if 'standard_xml' in item.keys():
        str_answer=item['standard_xml']
        item.pop('standard_xml')
    json_str = json.dumps(item, ensure_ascii=False, indent=2)
    # print(json_str)
    # input('aaaa')
    user_prompt = SamplePromptTemplate.prompt_template.replace("{INPUT_XML}", json_str).replace("{EXAMPLES}", str_example)
    return {'prompt':user_prompt,'response':str_answer}


def write_finetune_csv(json_path: str, csv_path: str) -> None:
    json_labels: List[Dict[str, Any]] = load_list_from_file(json_path)

    # Ensure output directory exists (optional but helpful)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
        writer.writeheader()
        for label_item in json_labels:
            result = build_data_for_fine_tuning(label_item)
            writer.writerow({
                "prompt": result["prompt"],
                "response": result["response"],
            })


fp_label_train='/home/hungphd/git/LearnJAX/doe/data-all/label-split/train.json'
fp_output_train='/home/hungphd/git/LearnJAX/doe/data-all/label-split/finetune_train.csv'
fp_label_valid='/home/hungphd/git/LearnJAX/doe/data-all/label-split/valid.json'
fp_output_valid='/home/hungphd/git/LearnJAX/doe/data-all/label-split/finetune_valid.csv'

# Generate both CSVs
write_finetune_csv(fp_label_train, fp_output_train)
write_finetune_csv(fp_label_valid, fp_output_valid)