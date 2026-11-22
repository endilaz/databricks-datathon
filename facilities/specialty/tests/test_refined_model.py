import unittest
import torch
from ranker.model import FeatureMLP, SemanticResidualMLP, model_from_checkpoint


class RefinedModelTests(unittest.TestCase):
    def model(self):
        # Symptom 0 and 1 share a specialty need; symptom 2 has another need.
        return SemanticResidualMLP(active_indices=[0,2],symptom_need_map=[[1,0],[1,0],[0,1]],dropout=0)

    def test_initial_score_is_positive_scaled_coverage(self):
        model=self.model();model.eval()
        p=torch.eye(3);h=torch.tensor([[1.,0,0],[1.,0,0],[0.,0,1.]])
        match=torch.tensor([[0.,0,.25,0,0],[0,0,.75,0,0],[0,0,1.,0,0]])
        parts=model.components(p,h,match)
        self.assertTrue(torch.equal(parts['learned_correction'],torch.zeros(3)))
        self.assertTrue(torch.allclose(parts['score'],8*match[:,2],atol=1e-5))

    def test_equal_need_profiles_share_representation_and_gradients_work(self):
        model=self.model();model.eval()
        p=torch.eye(3)[:2];h=torch.ones(2,3);match=torch.ones(2,5)*.5
        score=model(p,h,match)
        self.assertEqual(score[0].item(),score[1].item())
        score.sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None))

    def test_new_checkpoint_types_roundtrip(self):
        for kind,model in [('features_only',FeatureMLP()),('semantic_residual',self.model())]:
            copied=model_from_checkpoint({'architecture':kind,'config':model.config,'model_state':model.state_dict()})
            self.assertEqual(type(copied),type(model))


if __name__=='__main__':unittest.main()
