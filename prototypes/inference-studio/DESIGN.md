# Design decisions

Give the request population most of the space. Keep filters above the plots and table, retain the selected request in an adjacent inspector, and preserve selection while sorting or replaying. Show the filtered denominator and metric sample count. Display missing values as unavailable rather than zero. Use color plus text for failure and selection.

Use a single system sans family, tabular numerals, and monospace for identifiers and raw records. Dark surfaces use #090d12, #11171e, and #171e27; text uses #f0f3f6 and #9faab7; emerald #00e599 indicates selected or active controls. Dividers organize data without a grid of rounded cards. Charts share request scope with the table. Narrative diagnosis is outside this prototype.

All controls must have working outcomes. Native form controls and tables retain keyboard behavior. On narrower screens the inspector moves below the table; horizontal scrolling belongs to the table region. Errors, empty filters, missing metrics, warmups and all-failed cases are first-class states.
