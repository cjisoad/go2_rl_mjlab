import unittest


class ExtensionTests(unittest.TestCase):
    def test_resume_extends_existing_prefix_without_changing_finished_tasks(self):
        from scripts.train_sequence import TASKS, extend_saved_queue
        self.assertEqual(TASKS[-2:],("Unitree-Go2-Quad-Pedipulation","Unitree-Go2-Biped-Pedipulation"))
        old = [{"task":name,"status":"completed","reward":123.} for name in TASKS[:-2]]
        state = {"tasks":old.copy()}
        additions = [{"task":name,"max_iterations":rounds,"num_envs":4096,"status":"pending","completed_iterations":0}
                     for name,rounds in zip(TASKS[-2:],(5000,6000))]
        extend_saved_queue(state,additions)
        self.assertEqual(state["tasks"][:6],old)
        self.assertEqual(state["tasks"][6:],additions)
        extend_saved_queue(state,additions)
        self.assertEqual(len(state["tasks"]),8)

    def test_wrong_saved_order_is_rejected(self):
        from scripts.train_sequence import extend_saved_queue
        with self.assertRaises(ValueError):
            extend_saved_queue({"tasks":[{"task":"wrong"}]},[])

    def test_monitor_merges_planned_tasks_without_duplicates(self):
        from scripts.monitor_remote import merge_planned_tasks
        state={"tasks":[{"task":"old","status":"running"}]}
        planned={"tasks":[{"task":"new","status":"pending"}]}
        merge_planned_tasks(state,planned)
        merge_planned_tasks(state,planned)
        self.assertEqual([x["task"] for x in state["tasks"]],["old","new"])


    def test_handoff_waits_for_queue_lock_and_only_resumes_after_completion(self):
        import fcntl
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import time
        from textwrap import dedent
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            status=directory/"status.json"
            status.write_text(json.dumps({"tasks":[{"task":"old"}],"status":"running"}))
            marker=directory/"resumed.json"
            code=dedent("""
                import json,sys
                from pathlib import Path
                from scripts import extend_training_queue as extension
                directory=Path(sys.argv[1])
                extension.task_specs=lambda:[{"task":"old"},{"task":"new"}]
                extension.os.execv=lambda executable,args:(directory/"resumed.json").write_text(json.dumps(args))
                sys.argv=["extension","--state-dir",str(directory)]
                raise SystemExit(extension.main())
            """)
            with (directory/"queue.lock").open("w") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                child=subprocess.Popen([sys.executable,"-c",code,str(directory)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    deadline=time.monotonic()+10
                    while not (directory/"planned_tasks.json").exists() and child.poll() is None and time.monotonic()<deadline:
                        time.sleep(.02)
                    self.assertTrue((directory/"planned_tasks.json").exists())
                    self.assertIsNone(child.poll())
                    self.assertFalse(marker.exists())
                    status.write_text(json.dumps({"tasks":[{"task":"old","status":"completed"}],"status":"completed"}))
                    fcntl.flock(lock,fcntl.LOCK_UN)
                    output,error=child.communicate(timeout=10)
                    self.assertEqual(child.returncode,0,output+error)
                    self.assertIn("--resume",json.loads(marker.read_text()))
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.communicate()
