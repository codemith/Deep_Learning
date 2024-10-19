import sys
import torch
import json
from torch.utils.data import DataLoader
import pickle
import model
import Train
import bleu_eval

if __name__ == "__main__":
    test_data = sys.argv[1]
    test_json = "MLDS_hw2_1_data/testing_label.json"
    model_path = "Model/model1.pth"
    outputfile_path = sys.argv[2]

    # Load the model
    modelIP = torch.load(model_path)

    # Prepare the data
    files_dir = test_data
    i2w, w2i, dictonary = Train.dictonaryFunc(4)
    test_dataset = Train.test_dataloader(files_dir)

    # Using num_workers=0 to avoid multiprocessing issues on Mac
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # Run inference
    ss = Train.testfun(test_dataloader, modelIP, i2w)

    # Save the output
    try:
        with open(outputfile_path, 'w') as f:
            for id, s in ss:
                f.write('{},{}\n'.format(id, s))
            print('File updated successfully!')
    except FileNotFoundError:
        with open(outputfile_path, 'x') as f:
            for id, s in ss:
                f.write('{},{}\n'.format(id, s))
            print('File created and updated successfully!')

    # BLEU Evaluation
    test = json.load(open(test_json, 'r'))
    output = outputfile_path
    result = {}

    with open(output, 'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma + 1:]
            result[test_id] = caption

    bleu = []
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(bleu_eval.BLEU(result[item['id']], captions, True))
        bleu.append(score_per_video[0])

    average = sum(bleu) / len(bleu)
    print("Average BLEU score is " + str(average))
