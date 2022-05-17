.. _bug:

Bede User Group
---------------

The Bede User Group provides an interface between the Bede user community and the Bede Management Board. It provides an opportunity for Bede users to give feedback, request changes in policies and procedures, and interact directly with members of the Bede Support Group. 

More detailed information about the Bede User Group can be found in the
`Bede User Group Terms of Reference <https://n8cir.org.uk/supporting-research/facilities/bede/bug-tor/>`_ 

The Bede User Group meets once per term. Below is a list of issues raised at previous meetings, with the actions taken by the Bede Support Group, the Bede Management Board, or the Steering Group. In between meetings, issues can be raise in the Bede User Group channel on the Bede Slack workspace.

.. list-table:: Issues from Bede User Group Meetings
   :widths: 15 50 50
   :header-rows: 1

   * - Date
     - Issue raised
     - Actions
   * - 17.05.2022
     - Is it any easier to port ML applications on RHEL8 than previously?
     - Watson ML are no longer available, OpenCE is used instead, which so far works smoothly. `OpenCE use is documented here <https://bede-documentation.readthedocs.io/en/latest/software/applications/open-ce.html>`_ .     
   * - 08.02.2022
     - Relion and cryoEM modules need to be recompiled for RHEL8.
     - Action taken to contact IBM, who provided these modules.
   * - 
     - There are lots of single-node jobs on Bede, which might not be the intended use. However, users find it hard to run their jobs on multiple nodes with certain modules. It would be good if the BSG could do benchmarking work and provide jobscripts with the best configurations/parameters.
     - Action taken to discuss this in the BSG and consider a Benchmarking task force.
   * - 13.9.2021
     - It is hard to get enough resources on Bede for large scaling jobs (sometimes a week waiting time).
     - Two options discussed: Preemptive scheduling (would make checkpointing crucial in codes to avoid data loss when jobs are cancelled) and resource reservation. The latter seems to be the best way forward - implementation (email address or webform, frequency of available slots etc.) is put on the agenda for the next BSG meeting.
   * -
     - Per-default write permissions on folders of project members can lead to data loss and other issues.
     - Different levels of permissions on different storage areas explained. Discussion about changing the default added to the agenda of next BSG meeting.
  
   * - 12.4.2021
     - Gromacs installation and documentation. Installation and usage is a bit complicated, and more information would be appreciated.
     - Advise given in meeting re HECBioSim installation, further followed up per email. Documentation will be extended.
   * - 
     - Test queues for faster turnaround of test multi-node jobs needed. Multi-node jobs take a long time in the queue, makes development hard.
     - Recommended to use inference nodes, which are currently idle.

