**Added:**

* 'SNMFOptimizer.objective_log' attr: dictionary list to track the optimization
  process, recording the step, iteration, objective, and timestamp at each update.
  Uses the 'step', 'iteration', 'objective' and 'timestamp' keys.
* 'SNMFOptimizer(verbose : Optional[bool])' option and SNMFOptimizer.verbose
  attribute to allow users to toggle diagnostic console output.

**Changed:**

* Modified all print messages for improved readability and tied them to the new
  verbose flag.
* Refactored convergence checks and step-size calculations to pull objective
  values directly from objective_log instead of relying on a separate history
  array.

**Deprecated:**

* <news item>

**Removed:**

* Removed the 'SNMFOptimizer._objective_history' list, which was made redundant
  by the comprehensive 'SNMFOptimizer.objective_log' tracking system.

**Fixed:**

* <news item>

**Security:**

* <news item>
