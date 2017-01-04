#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This program try to slove the following problem:
# Assume the map is A A1 A2 A3 A4
# The input is A B C D
# The output is A1 B C D \n A2 B C D \n A3 B C D \n A4 B C D \n
# This is a one2multiple mapping.

# Attentation: Use ":" to join the post-map. 


if (@ARGV > 0 && $ARGV[0] eq "-f") {
  shift @ARGV;
  $field_spec = shift @ARGV; 
  if ($field_spec =~ m/^\d+$/) {
    $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
  }
  if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
    if ($1 ne "") {
      $field_begin = $1 - 1;    # Change to zero-based indexing.
    }
    if ($2 ne "") {
      $field_end = $2 - 1;      # Change to zero-based indexing.
    }
  }
  if (!defined $field_begin && !defined $field_end) {
    die "Bad argument to -f option: $field_spec"; 
  }
}

# Mapping is obligatory
$permissive = 0;
if (@ARGV > 0 && $ARGV[0] eq '--permissive') {
  shift @ARGV;
  # Mapping is optional (missing key is printed to output)
  $permissive = 1;
}

if(@ARGV != 1) {
  print STDERR "Invalid usage: " . join(" ", @ARGV) . "\n";
  print STDERR "Usage: apply_map_one2mult.pl [options] map <input >output\n" .
    "options: [-f <field-range> ]\n" .
    "Applies the map 'map' to all input text, where each line of the map\n" .
    "is interpreted as a map from the first field to the list of the other fields\n" .
    "Note: <field-range> can look like 4-5, or 4-, or 5-, or 1, it means the field\n" .
    "range in the input to apply the map to.\n" .
    "e.g.: echo A B | apply_map.pl a.txt\n" .
    "where a.txt is:\n" .
    "A A1 A2\n" .
    "B B1\n" .
    "will produce:\n" .
    "A1 B1\n" .
    "A2 B1\n";
  exit(1);
}

($map) = @ARGV;
open(M, "<$map") || die "Error opening map file $map: $!";

while (<M>) {
  @A = split(" ", $_);
  @A >= 1 || die "apply_map.pl: empty line.";
  $i = shift @A;
  $o = join(":", @A);
  $map{$i} = $o;
}

sub printcontent {
  (my $start, my @string)=@_;

  if ( $start == @string ) { print join(" ",@string) . "\n";
  } else {
    my $tmp = $string[$start];
    my @Word = split(":", $tmp);
    if ( @Word != 1) {
      foreach(@Word) {
        $string[$start] = $_;
        $start++;
        &printcontent($start, @string);
        $start--;
      }
    } else {
      $start++;
      &printcontent($start, @string);
    }
  }
}

while(<STDIN>) {
  @A = split(" ", $_);
  for ($x = 0; $x < @A; $x++) {
    if ( (!defined $field_begin || $x >= $field_begin)
         && (!defined $field_end || $x <= $field_end)) {
      $a = $A[$x];
      if (!defined $map{$a}) {
        if (!$permissive) {
          die "apply_map.pl: undefined key $a\n"; 
        } else {
          print STDERR "apply_map.pl: warning! missing key $a\n";
        }
      } else {
        $A[$x] = $map{$a}; 
      }
    }
  }
  # print the content
  &printcontent(0,@A);
}
