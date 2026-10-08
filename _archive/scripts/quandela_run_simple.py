import perceval as pcvl
import perceval.providers.quandela as qnd
import time

# 1. Define your Quandela Cloud API Token
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mjg3MiwiZXhwIjoxNzg2OTgyODkxLjA2MTI0NH0.hxaFfc_BDS5euV9NOUuerUyFAg5GmpW7xw87zLKvSPzqJlwDyV1rBMopB0MVPXCjDtn5Uy6JlXSh6ivl31Eghg"

# 2. Establish a session with a designated cloud backend
# Options include "sim:slos" (simulator) or physical QPUs like "qpu:ascella"
session = qnd.Session(platform_name="sim:slos", token=TOKEN)
remote_processor = session.build_remote_processor()

# 3. Create a simple Photonic Quantum Circuit (e.g., a Beam Splitter)
circuit = pcvl.Circuit(2)
circuit.add(0, pcvl.BS()) # Adds a beam splitter on modes 0 and 1

# 4. Configure the processor with the circuit and input state
remote_processor.set_circuit(circuit)
remote_processor.with_input(pcvl.BasicState([1, 1])) # Input 1 photon in each mode
remote_processor.min_detected_photons_filter(2)

# 5. Submit the job to the cloud to take samples
# In Perceval 1.2+, remote sampling is submitted through a Sampler job.
sampler = pcvl.algorithm.Sampler(remote_processor, max_shots_per_call=1000)
job = sampler.sample_count.execute_async(1000)

print(f"Job submitted successfully! Job ID: {job.id}")

# 6. Retrieve and display the results
# This will block execution until the cloud finishes processing
while not job.is_complete:
	time.sleep(1)

results = job.get_results()
print("Sampling Results:")
print(results['results'])

# Resume a previously submitted job using its unique ID
RESUME_JOB_ID = ""
if RESUME_JOB_ID:
	remote_job = remote_processor.resume_job(RESUME_JOB_ID)
	while not remote_job.is_complete:
		time.sleep(1)
	results = remote_job.get_results()
	print("Resumed Job Results:")
	print(results['results'])
