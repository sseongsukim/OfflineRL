from agents.rql import RQLAgent
from agents.sharsa import SHARSAAgent
from agents.trl import TRLAgent
from agents.fql import FQLAgent

from agents.crl import CRLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent

from agents.sac import SACAgent

agents = dict(
    sac=SACAgent,
    qrl=QRLAgent,
    hiql=HIQLAgent,
    crl=CRLAgent,
    fql=FQLAgent,
    trl=TRLAgent,
    sharsa=SHARSAAgent,
    rql=RQLAgent,
)
