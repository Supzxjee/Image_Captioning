"""Same CSA parameters/decoder; repeat encoded memories only after image encoding."""
from csa_patch_model import CSAPatchCaptioner


class SharedCSAPatchCaptioner(CSAPatchCaptioner):
    def forward(self, patches, prefix, p, inverse=None):
        memory = self.encode(patches, p)
        if inverse is not None:
            memory = memory.index_select(0, inverse)
        return self.decode(memory, prefix)
