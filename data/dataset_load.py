import glob
import os
import re

import datasets

_CITATION = """@misc{alon2019code2seq,
      title={code2seq: Generating Sequences from Structured Representations of Code}, 
      author={Uri Alon and Shaked Brody and Omer Levy and Eran Yahav},
      year={2019},
      eprint={1808.01400},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}"""
_DESCRIPTION = """Contains 11 relatively large Java projects, originally used for 11 distinct models for training and predicting within the scope of the same project (Allamanis et al., 2016). This dataset contains about 700K examples."""
_URL = "https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz"
_HOMEPAGE = "https://paperswithcode.com/dataset/code2seq"
_LICENSE = ""


class JavaSmall(datasets.GeneratorBasedBuilder):
    """Java-small dataset."""

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "filename": datasets.Value("string")
                # These are the features of your dataset like images, labels ...
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"{data_dir}/java-small/training/**/*.java",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": f"{data_dir}/java-small/test/**/*.java",
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": f"{data_dir}/java-small/validation/**/*.java",
                    "split": "validation"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        datasetlist = glob.glob(filepath, recursive=True)
        for key, filename in enumerate(datasetlist):
            if "TipoMensagem.java" in filename:
                continue
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
            code = self._remove_comments(code)
            code = self._remove_empty_lines(code)
            yield key, {
                "text": code,
                "filename": filename
            }

    def _remove_comments(self, string):
        pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
        # first group captures quoted strings (double or single)
        # second group captures comments (//single-line or /* multi-line */)
        regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

        def _replacer(match):
            # if the 2nd group (capturing comments) is not None,
            # it means we have captured a non-quoted (real) comment string.
            if match.group(2) is not None:
                return ""  # so we will return empty to remove the comment
            else:  # otherwise, we will return the 1st group
                return match.group(1)  # captured quoted-string

        return regex.sub(_replacer, string)

    def _remove_empty_lines(self, string):
        filtered = filter(lambda x: not re.match(r'^\s*$', x), string.split('\n'))
        return '\n'.join(filtered)
