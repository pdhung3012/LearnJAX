class BasicPromptTemplate:
    str_query="""You are an expert in Pytorch to JAX translation, please translate the given code snippet in Pytorch into JAX:
    
    '''
    {PYTORCH_CODE}
    '''
    
    Please return only the JAX code as output.

    """

    str_answer='{JAX_CODE}'