from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class VectorStoreInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        # connection to vector store 
        pass

    @abstractmethod
    def search(self,
        query_embedding: List[float],
        top_k: int,
        filter: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        # search for nearest neighbors 
        # :param query_embedding: The vector embedding to query
        # :param top_k: Number of results to return
        # :param filter: Optional filter on metadata
        # :return: List of tuples (id, score, metadata)
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        # delete vectors by their IDs
        # :param ids: List of IDs to remove
        pass
    
    @abstractmethod
    def close(self) -> None:
        # close connection and clean up resources
        pass


    
