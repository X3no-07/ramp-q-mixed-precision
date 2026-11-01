"""Exact22feature implementation with fixed128row IoU blocks.

No weights, candidates, labels, scores, threshold or preprocessing is changed.
"""
import numpy as np
import gate1_frozen_box_scorer as S

BLOCK_ROWS=128


def visible_features(raw,predictions):
    n=len(predictions)
    if not n:return np.empty((0,len(S.FEATURES)),np.float64)
    ids=np.array([d['raw_anchor_index'] for d in predictions],np.int64)
    S.require(raw.ndim==3 and raw.shape[0:2]==(1,14) and np.isfinite(raw).all(),'Raw domain differs')
    S.require(len(set(ids.tolist()))==n,'Duplicate raw anchor')
    scores=raw[0,4:][:,ids].T.astype(np.float64)
    S.require(scores.shape==(n,10),'Class score indexing differs')
    cls=np.array([d['category']-1 for d in predictions])
    S.require(np.array_equal(scores.argmax(1),cls),'Fixed candidate category differs')
    maximum=scores.max(1);second=np.partition(scores,-2,axis=1)[:,-2]
    S.require(np.array_equal(maximum,np.array([d['score'] for d in predictions])),'Candidate score differs')
    distribution=scores/np.maximum(scores.sum(1,keepdims=True),1e-12)
    entropy=-(distribution*np.log(np.maximum(distribution,1e-12))).sum(1)/np.log(10)
    boxes=raw[0,:4][:,ids].T.astype(np.float64)
    S.require(boxes.shape==(n,4) and (boxes[:,2:]>0).all(),'Fixed candidate geometry differs')
    corners=np.column_stack((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2))
    area=np.maximum(0,corners[:,2:]-corners[:,:2]).prod(1)
    neighbors=np.empty((n,4),np.float64);indices=np.arange(n)
    for start in range(0,n,BLOCK_ROWS):
        end=min(start+BLOCK_ROWS,n);b=corners[start:end]
        width=np.maximum(0,np.minimum(b[:,None,2],corners[None,:,2])-np.maximum(b[:,None,0],corners[None,:,0]))
        height=np.maximum(0,np.minimum(b[:,None,3],corners[None,:,3])-np.maximum(b[:,None,1],corners[None,:,1]))
        intersection=width*height
        overlap=intersection/np.maximum(area[start:end,None]+area[None,:]-intersection,1e-12)
        overlap[np.arange(end-start),indices[start:end]]=0
        same=cls[start:end,None]==cls[None,:]
        close=same&(overlap>=.5)
        higher=(maximum[None,:]>maximum[start:end,None])|((maximum[None,:]==maximum[start:end,None])&(indices[None,:]<indices[start:end,None]))
        neighbors[start:end]=np.column_stack((np.log1p(close.sum(1)),np.where(same&higher,overlap,0).max(1),
            np.where(~same,overlap,0).max(1),np.where(close,maximum[None,:],0).max(1)))
    out=np.column_stack((S.logit(maximum),S.logit(second),maximum-second,entropy,
        np.log(np.maximum(boxes[:,2]/640,1e-6)),np.log(np.maximum(boxes[:,3]/640,1e-6)),boxes[:,0]/640,boxes[:,1]/640,
        neighbors,np.eye(10)[cls]))
    out=np.clip(out,-16,16)
    S.require(out.shape==(n,len(S.FEATURES)) and np.isfinite(out).all(),'Feature shape/finite differs')
    return out
