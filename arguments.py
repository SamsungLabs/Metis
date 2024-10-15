# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = _add_model_args(parser)
    parser = _add_gpt_model_args(parser)
    parser = _add_cluster_args(parser)
    parser = _add_hetspeed_args(parser)
    parser = _add_env_args(parser)
    args = parser.parse_args()
    return args


def _add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_size', type=str)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--gbs', type=int)
    return parser

def _add_gpt_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--sequence_length', type=int)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--attention_head_size', type=int)
    return parser

def _add_cluster_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--hostfile_path')
    parser.add_argument('--clusterfile_path')
    return parser


def _add_env_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--log_path')
    parser.add_argument('--home_dir')
    return parser


def _add_hetspeed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--profile_data_path')
    parser.add_argument('--max_profiled_tp_degree', type=int)
    parser.add_argument('--max_profiled_batch_size', type=int)
    parser.add_argument('--min_group_scale_variance', type=int)
    parser.add_argument('--max_permute_len', type=int)

    return parser
