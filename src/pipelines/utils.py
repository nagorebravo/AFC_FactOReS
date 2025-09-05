import torch


@torch.no_grad()
def cosine_similarity(
    query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute the similarity between each query and passage embeddings.

    Args:
        query_embeddings (torch.Tensor): The query embeddings.
        passage_embeddings (torch.Tensor): The passage embeddings.
    Returns:
        torch.Tensor: The similarity between each query and passage embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embeddings = query_embeddings.to(device)
    passage_embeddings = passage_embeddings.to(device)

    similarity = torch.mm(query_embeddings, passage_embeddings.T)

    return similarity.cpu()
