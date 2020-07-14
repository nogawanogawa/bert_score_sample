from bert_score import score

def calc_bert_score(cands, refs):
    """ BERTスコアの算出

    Args:
        cands ([List[str]]): [比較元の文]
        refs ([List[str]]): [比較対象の文]

    Returns:
        [(List[float], List[float], List[float])]: [(Precision, Recall, F1スコア)]
    """
    Precision, Recall, F1 = score(cands, refs, lang="ja", verbose=True)
    return Precision.numpy().tolist(), Recall.numpy().tolist(), F1.numpy().tolist()


if __name__ == "__main__":
    """ サンプル実行 """ 
    with open("hyps.txt") as f:
        cands = [line.strip() for line in f]

    with open("refs.txt") as f:
        refs = [line.strip() for line in f]
    
    P, R, F1 = calc_bert_score(cands, refs)
    for p,r, f1 in zip(P, R, F1):
        print("P:%f, R:%f, F1:%f" %(p, r, f1))