import re
from typing import Dict, List, NamedTuple


class Metadata(NamedTuple):
    source: str
    url: str
    favicon: str


class CitationManager:
    def __init__(self, metadatas: List[Dict[str, str]] = None):
        self.metadatas: List[Metadata] = []
        self.url2id: Dict[str, int] = {}
        self.id2metadata: Dict[int, Metadata] = {}

        if metadatas:
            self.add_metadatas(metadatas)

    def add_metadatas(self, metadatas: List[Dict[str, str]]) -> List[int]:
        new_ids = []
        for metadata in metadatas:
            metadata_obj = Metadata(**metadata)
            if metadata_obj.url not in self.url2id:
                new_id = len(self.metadatas)
                self.metadatas.append(metadata_obj)
                self.url2id[metadata_obj.url] = new_id
                self.id2metadata[new_id] = metadata_obj
                new_ids.append(new_id)
            else:
                new_ids.append(self.url2id[metadata_obj.url])
        return new_ids

    def get_ids(self, urls: List[str]) -> List[int]:
        return [self.url2id[url] for url in urls if url in self.url2id]

    def get_metadata(self, ids: List[int]) -> List[Metadata]:
        return [self.id2metadata[id] for id in ids if id in self.id2metadata]

    @staticmethod
    def format_prompt(id: int, text: str) -> str:
        return f"[{id}] {' '.join(text.split())}"

    def prepare_for_prompt(
        self, texts: List[str], metadatas: List[Dict[str, str]]
    ) -> str:
        ids = self.add_metadatas(metadatas)
        formatted_texts = []
        for text, id in zip(texts, ids):
            formatted_text = self.format_prompt(id, text)
            if formatted_text not in formatted_texts:
                formatted_texts.append(formatted_text)
        return "\n".join(sorted(formatted_texts, key=lambda x: int(x.split(']')[0].strip('['))))

    @staticmethod
    def retrieve_ids_from_text(text: str) -> List[int]:
        return sorted(
            set(
                int(num)
                for citation in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
                for num in citation.split(",")
            )
        )

    def get_html_citations(self, ids: List[int]) -> List[str]:
        return [
            f'<a href="{metadata.url}" style="display:inline-flex;align-items:center;background-color:#4CAF50;color:white;padding:8px 12px;border-radius:12px;text-decoration:none;font-weight:bold;margin:5px;font-size:14px;">'
            f"[{id}] {metadata.source} "
            f'<img src="{metadata.favicon}" alt="{metadata.source}" style="width:16px;height:16px;margin-left:8px;">'
            f"</a>"
            for id, metadata in zip(ids, self.get_metadata(ids))
        ]

    def reorder_citations(self, texts: List[str]) -> tuple[List[str], List[int]]:
        original_id2metadata = self.id2metadata.copy()

        citation_order = [
            int(num)
            for text in texts
            for citation in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
            for num in citation.split(",")
            if int(num) in original_id2metadata  # Only include valid citation IDs
        ]

        # Create a new ordering for the citations
        reorder_dict = {id: i + 1 for i, id in enumerate(dict.fromkeys(citation_order))}

        # Include any remaining valid IDs that weren't in the texts
        for id in original_id2metadata:
            if id not in reorder_dict:
                reorder_dict[id] = len(reorder_dict) + 1

        # Update the internal mappings
        self.id2metadata = {
            reorder_dict[id]: metadata for id, metadata in original_id2metadata.items()
        }
        self.url2id = {metadata.url: id for id, metadata in self.id2metadata.items()}

        # Update the citations in the texts
        cited_ids = set()
        for i, text in enumerate(texts):

            def replace_citation(match):
                citation_content = match.group(2).strip("[]")
                old_ids = [
                    int(num.strip())
                    for num in citation_content.split(",")
                    if num.strip()
                    and num.strip().isdigit()
                    and int(num.strip()) in original_id2metadata
                ]
                pre_space = " " if match.group("pre") else ""
                post_space = " " if match.group("post") else ""
                if not old_ids:
                    return pre_space + post_space  # Remove empty citations
                new_ids = [reorder_dict[old_id] for old_id in old_ids]
                cited_ids.update(new_ids)
                return f"{pre_space}[{','.join(map(str, new_ids))}]{post_space}"

            # Replace citations, preserving newlines and spaces
            lines = text.split("\n")
            for j, line in enumerate(lines):
                lines[j] = re.sub(
                    r"(?P<pre>\s*)(\[(\d+(?:,\s*\d+)*)\])(?P<post>\s*)",
                    replace_citation,
                    line,
                )
                # Remove any resulting double spaces within each line
                lines[j] = re.sub(r"\s{2,}", " ", lines[j]).strip()

            texts[i] = "\n".join(lines)

        return texts, sorted(cited_ids)
