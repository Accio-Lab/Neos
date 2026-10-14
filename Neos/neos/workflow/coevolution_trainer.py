
from __future__ import annotations
import concurrent.futures
import logging
import os
import time
from pathlib import Path
from neos.workflow.coevolution import atomic_json, read_json, build_mei_batch, append_event
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync

logger=logging.getLogger(__name__)

class CoevolutionDriver:
    def __init__(self, trainer):
        self.trainer=trainer
        self.root=Path(os.environ["NEOS_COEV_ROOT"])
        self.k=int(os.getenv("NEOS_COEV_K","10"))
        if self.k<1:raise ValueError("MEI update interval must be positive")
        self.pool=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix="coev-dataset")
        self.pending=None
        self.mei_model=None
        self.mei_version=0
        self.last_requested_update=0

    def policy_state(self,phase,step,**values):
        atomic_json(self.root/"policy.json",{
            "phase":phase,"step":int(step),"mei_model":self.mei_model,
            "mei_version":self.mei_version,"at_unix":time.time(),**values})

    def before_rollout(self,step):
        required=((int(step)-1)//self.k)*self.k
        self.policy_state("request_rollout",step,required_mei_after_step=required)
        started=time.monotonic()
        timeout=float(os.getenv("NEOS_COEV_WAIT_TIMEOUT","21600"))
        while True:
            if self.pending is not None and self.pending.done():self.pending.result()
            status=read_json(self.root/"serving.json",{})
            if status.get("status")=="failed":
                raise RuntimeError("MEI update failed: "+str(status.get("error")))
            if (status.get("status")=="ready" and int(status.get("policy_step",0))==step
                and int(status.get("after_policy_step",-1))>=required):
                self.mei_model=status["model"];self.mei_version=int(status["version"])
                break
            if time.monotonic()-started>timeout:raise TimeoutError("Waiting for the next MEI version timed out")
            time.sleep(1)
        self.policy_state("rollout",step)
        append_event("lifecycle",{"event":"formal_rollout_begin","step":step,
            "mei_model":self.mei_model,"wait_seconds":time.monotonic()-started})

    def selected(self,step):
        self.policy_state("policy_update",step)
        append_event("lifecycle",{"event":"selected_batch_credit_ready","step":step,
                                 "mei_model":self.mei_model})
        if step % self.k:return
        if self.pending is not None:self.pending.result()
        self.last_requested_update=step
        first=step-self.k+1
        old_model=self.mei_model;old_version=self.mei_version
        def prepare():
            output=self.root/"mei_training"/f"after-step-{step}"/"data"
            for attempt in range(3):
                manifest=build_mei_batch(self.root,first_step=first,last_step=step,
                                           mei_model=old_model,output=output)
                if not manifest["not_yet_published_trace_ids"]:break
                if attempt<2:time.sleep(2)
            if manifest["not_yet_published_trace_ids"]:
                append_event("lifecycle",{"event":"incomplete_mei_examples_excluded",
                    "step":step,"count":manifest["not_yet_published_trace_ids"],
                    "audit":str(output/"selection.jsonl")})
            epochs=float(os.getenv("NEOS_COEV_MEI_EPOCHS","1"))
            if not epochs.is_integer() or epochs<1:
                raise ValueError("MEI SFT requires a positive integer epoch count")
            job={"after_policy_step":step,"source_mei_model":old_model,
                 "source_mei_version":old_version,"data":manifest,
                 "learning_rate":float(os.getenv("NEOS_COEV_MEI_LR","1e-6")),
                 "epochs":int(epochs),"at_unix":time.time()}
            atomic_json(self.root/"jobs"/f"after-step-{step:06d}.json",job)
            append_event("lifecycle",{"event":"mei_sft_job_published","step":step,
                                     "train_rows":manifest["train_rows"]})
            return job
        self.pending=self.pool.submit(prepare)

    def finish(self,step):
        if self.pending is not None:self.pending.result()
        self.policy_state("waiting_final_mei",step,required_mei_after_step=self.last_requested_update)
        started=time.monotonic()
        while self.last_requested_update:
            status=read_json(self.root/"serving.json",{})
            if status.get("status")=="failed":raise RuntimeError(str(status.get("error")))
            if (int(status.get("after_policy_step",-1))>=self.last_requested_update
                and status.get("status") in {"ready","waiting_policy"}):break
            if time.monotonic()-started>float(os.getenv("NEOS_COEV_WAIT_TIMEOUT","21600")):
                raise TimeoutError("Final MEI update timed out")
            time.sleep(1)
        self.policy_state("complete",step)
        self.pool.shutdown(wait=True)

    def failed(self,error):
        self.policy_state("failed",self.trainer.global_steps,error=str(error)[:2000])
        self.pool.shutdown(wait=False,cancel_futures=True)

class PPOTrainerCoevolution(PPOTrainerColocateAsync):
    pass
    def on_train_begin(self):
        self.coevolution=CoevolutionDriver(self)
        return super().on_train_begin()

    def _adaptive_submit_round(self,batch,*,rollout_step):
        self.coevolution.before_rollout(rollout_step)
        return super()._adaptive_submit_round(batch,rollout_step=rollout_step)

    def on_sample_end(self):
        self.coevolution.policy_state("policy_update",self.global_steps)
        return super().on_sample_end()

    def _compute_advantage(self,batch,metrics):
        result=super()._compute_advantage(batch,metrics)
        
        
        
        import transfer_queue as tq
        from neos.workflow.coevolution import observe_answer
        from neos.workflow.turn_credit import _unwrap_mapping
        data=tq.kv_batch_get(keys=batch.keys,partition_id=batch.partition_id,select_fields=['extra_fields'])
        raw=data.get('extra_fields')
        if raw is not None:
            rows=raw.tolist() if hasattr(raw,'tolist') else list(raw)
            for value in rows:
                fields=_unwrap_mapping(value)
                reward=_unwrap_mapping(fields.get('reward_extra_info'))
                observe_answer(fields,correct=float(reward.get('acc') or 0)>0.5,
                               judge_called=bool(reward.get('judge_called')),
                               judge_parse_failed=bool(reward.get('judge_parse_failed')))
        
        
        self.coevolution.selected(self.global_steps)
        return result

    def on_step_end(self):
        result=super().on_step_end()
        if self.global_steps>=self.total_training_steps:
            self.coevolution.finish(self.global_steps)
        return result

    def fit(self,agent_loop_manager):
        try:
            return super().fit(agent_loop_manager)
        except BaseException as exc:
            if getattr(self,"coevolution",None):self.coevolution.failed(exc)
            raise
