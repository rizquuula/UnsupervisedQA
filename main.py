from unsupervisedqa.generate_synthetic_qa_data import generate_synthetic_training_data

class Args:
    input_file = 'test000.txt' # input file, see readme for formatting info
    output_file = 'output000' # Path to write generated data to, see readme for formatting info
    input_file_format = 'txt' # input file format, see readme for more info, default is txt
    output_file_formats = 'jsonl' # comma-seperated list of output file formats, from [jsonl, squad],
    translation_method = 'identity' # define the method to generate clozes -- either the Unsupervised NMT method (unmt)
    use_named_entity_clozes = True # pass this flag to use named entity answer prior instead of noun phrases (recommended for downstream performance) 
    use_subclause_clozes = False # pass this flag to shorten clozes with constituency parsing instead of using sentence boundaries (recommended for downstream performance)
    use_wh_heuristic = True # pass this flag to use the wh-word heuristic (recommended for downstream performance). Only compatable with named entity clozes

args = Args()
generate_synthetic_training_data(args)
