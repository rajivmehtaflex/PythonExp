
"""
2021.07.16

For encoders that already fine-tune on the targets (namely text)
the unity mixer just arg-maxes the output of the encoder.
"""

from typing import List

import torch
import pandas as pd

from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs
from loguru import logger
from ttictoc import tic,toc

from txtai.pipeline import Labels


class FetchDB(BaseMixer):
    def __init__(self, stop_after: float, dtype_dict: dict, target: str, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        self.supports_proba = False
        self.stable = True
        self.labels = Labels()
        logger.add("/home/gitpod/out.log")

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info("Unit Mixer just borrows from encoder")

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        if args.predict_proba:
            # @TODO: depending on the target encoder, this might be enabled
            log.warning('This model does not output probability estimates')

        decoded_predictions: List[object] = []
        tags = ['neutral', 'positive', 'negative']
        logger.info(f'data -->{ConcatedEncodedDs([ds]).get_column_original_data("text").tolist()}') 
        
        for text in ConcatedEncodedDs([ds]).get_column_original_data("text").tolist():
            result=tags[self.labels(text, tags)[0][0]] 
            logger.info(f'result -->{result}')
            decoded_predictions.extend([result])

        ydf = pd.DataFrame({"prediction": decoded_predictions})
        return ydf