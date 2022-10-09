import torch
import operator
import transformers
import pandas as pd
from pathlib import Path
from typeguard import typechecked

from typing import Optional, Any, Union, Dict

from ru_encoders_benchmark.tasks.task import Task

@typechecked
class SemanticTextualSimilarityTask(Task):
    def __init__(self, filename_dict: Dict[str, str] = {'train': 'stsb_multi_mt_train.csv',
                                                        'test': 'stsb_multi_mt_test.csv',
                                                        'dev': 'stsb_multi_mt_dev.csv'}):
        super().__init__("SemanticTextualSimilarityTask")
        self.data = {}
        for key in filename_dict.keys():
            self.data[key] = self._get_data(filename_dict.get(key))
            self._logger.info(f"Dataset '{key}' for {self.task_name} task loaded")
        self.full_cache = {}
        self.score_cache = {}

    def _get_data(self, filename: Union[str, Path],
                  path: Union[str, Path] = None):
        if path is None:
            path = LM_TASK2PATH[self.task_name]
        try:
            df = pd.read_csv(str(find_file(filename, path)),
                             index_col=0)
        except Exception as e:
            self._logger.error(f"Error during train data loading: {e}")
            return None
        return df

    def eval(self,
             embedder: torch.nn.Module,
             tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
             model_name: str,
             data_split_name: str = 'test'):
        """
        STS benchmark is formulated as regression task.
        """
        try:
            results = eval_text_pairs_regression(
                embedding_func=lambda text: generate_embeddings(text,
                                                                embedder,
                                                                tokenizer,
                                                                embedding_func=get_both_embeddings,
                                                                return_attention_mask=True
                                                                ),
                data=self.data.get(data_split_name),
                text1='sentence1',
                text2='sentence2',
                target_col='similarity_score',
                comparing_op=operator.mul
            )
            max_score = max(results.values())
            self.full_cache[model_name] = results
            self.score_cache[model_name] = max_score
        except Exception as e:
            self._logger.error(f"Error occured during validation of {model_name} model"
                               f"on task {self.task_name}:\n{e}")
            return None, {}
        return max_score, results
