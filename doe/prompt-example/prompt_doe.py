
class InContextPromptTemplate:
    system_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input JSON following a specified requirement defined in corresponding a JSON file. We will give you:
1. Input non-standard JSON string.
2. A JSON file as information about the schema of output XML you should generate. It will be in JSON array, each item has 'page' as page number and 'text' as text content of that page.
Your task is to provide the output:
3. Generate the standardize XML from the input as non-standard XML following the JSON file in the context.
Return the XML file only.        
    '''
    prompt_template='''
Translating this input to standard XML file following a context file (2 inputs).
1. Input JSON:
"""
{input_XML}
"""

2. JSON as XML schema: please check the context

3. Please provide the output as standard XML and return the XML as output only.
'''

class SamplePromptTemplate:
    system_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input JSON, given the context from one to several examples as input/ expected output of the problem. We will give you:
1. Input non-standard JSON string.
2. Examples of expected input and output in context (2.1. Example 1. - Input: - Output).
Your task is to provide the output:
3. Generate the standardize XML from the input as non-standard XML following the JSON file in the context.
Return the XML file only.        
    '''
    prompt_template='''
Translating this input to standard XML file following a context file (2 inputs).
1. Input JSON:
"""
{input_XML}
"""

2. Examples of input and output.
{examples}

3. Please provide the output as standard XML and return the XML as output only.
'''

class FineTuningPromptTemplate:
    system_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input JSON following a specified requirement defined in corresponding a JSON file. We will give you:
1. Input non-standard JSON string.
2. A JSON file as information about the schema of output XML you should generate. It will be in JSON array, each item has 'page' as page number and 'text' as text content of that page.
Your task is to provide the output:
3. Generate the standardize XML from the input as non-standard XML following the JSON file in the context.
Return the XML file only.        
    '''
    prompt_template='''
Translating this input to standard XML file following a context file (2 inputs).
1. Input JSON string:
"""
{input_XML}
"""

2. JSON as XML schema: please check the context

3. Please provide the output as standard XML and return the XML as output only.
'''

class DPOPromptTemplate:
    system_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input JSON following a specified requirement defined in corresponding a JSON file. We will give you:
1. Input non-standard JSON string.
2. A JSON file as information about the schema of output XML you should generate. It will be in JSON array, each item has 'page' as page number and 'text' as text content of that page.
Your task is to provide the output:
3. Generate the standardize XML from the input as non-standard XML following the JSON file in the context.
Return the XML file only.        
    '''
    prompt_template='''
Translating this input to standard XML file following a context file (2 inputs).
1. Input JSON string:
"""
{input_XML}
"""

2. JSON as XML schema: please check the context

3. Please provide the output as standard XML and return the XML as output only.
'''


class LabelGenerationPromptTemplate:
    system_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input json text. The schema of standard XML is defined in a public PDF file published on Github.
    '''
    prompt_template='''
Translating this input to standard XML file following a context file (2 inputs).
1. Input JSON string:
"""
{INPUT_XML}
"""

2. The standard XML schema is specified in:
"""
{SCHEMA_URL}
"""

3. Please provide the output as standard XML and return the XML as output only.
'''